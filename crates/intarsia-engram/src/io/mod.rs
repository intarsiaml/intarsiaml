//! Raw read backends: io_uring (Linux) and a portable pread pool, both on O_DIRECT descriptors.
//!
//! A backend moves [`ReadOp`]s (aligned offset and length, into an aligned slot) and hands back completions
//! tagged with the caller's `tag`. It knows nothing about rows; [`crate::store`] maps rows to ops.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::path::Path;
use std::time::Duration;

pub mod pool;
#[cfg(target_os = "linux")]
pub mod uring;

/// Alignment of every buffer handed to the kernel (covers 512 B and 4 KiB logical blocks).
pub const BUF_ALIGN: usize = 4096;

/// A zeroed, `BUF_ALIGN`-aligned heap buffer whose address never moves (the kernel writes into it).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain owned memory; synchronisation of concurrent kernel/worker writes is the users' job (each slot
// is written by exactly one in-flight op and read only after its completion).
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = len.max(BUF_ALIGN).next_multiple_of(BUF_ALIGN);
        let layout = Layout::from_size_align(len, BUF_ALIGN).expect("layout");
        // SAFETY: non-zero size, valid alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "out of memory allocating {len} B aligned buffer");
        Self { ptr, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: owned allocation of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: owned allocation of `len` bytes, unique borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.ptr, Layout::from_size_align(self.len, BUF_ALIGN).unwrap()) }
    }
}

/// A file opened for table reads: O_DIRECT on Linux (falls back to buffered where the file system refuses it),
/// `F_NOCACHE` on macOS. `direct` says what was obtained.
pub struct DirectFile {
    pub file: File,
    pub direct: bool,
}

impl DirectFile {
    pub fn open(path: &Path, want_direct: bool) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            if want_direct {
                match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
                    Ok(file) => return Ok(Self { file, direct: true }),
                    Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {}
                    Err(e) => return Err(e),
                }
            }
            let file = File::open(path)?;
            if !want_direct {
                // a bench's buffered run reads random rows: no readahead
                // SAFETY: valid fd.
                unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
            }
            Ok(Self { file, direct: false })
        }
        #[cfg(target_os = "macos")]
        {
            let file = File::open(path)?;
            let mut direct = false;
            if want_direct {
                // SAFETY: valid fd.
                direct = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == 0;
            }
            Ok(Self { file, direct })
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = want_direct;
            Ok(Self { file: File::open(path)?, direct: false })
        }
    }

    pub fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Drop this file's pages from the page cache (after a buffered bench run).
    pub fn drop_cache(&self) {
        #[cfg(target_os = "linux")]
        // SAFETY: valid fd.
        unsafe {
            libc::posix_fadvise(self.fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

/// `pread` until `buf` is full; error on EOF.
pub fn pread_full(fd: RawFd, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        // SAFETY: buf[done..] is valid writable memory.
        let n = unsafe {
            libc::pread(fd, buf[done..].as_mut_ptr().cast(), buf.len() - done, (offset + done as u64) as libc::off_t)
        };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past end of table file"));
        }
        done += n as usize;
    }
    Ok(())
}

/// Read `out.len()` bytes at any `offset` through a (possibly O_DIRECT) descriptor via aligned bounce reads of
/// up to `chunk` bytes. Used for headers and the resident scale array, so nothing lands in the page cache.
pub fn read_range(f: &DirectFile, offset: u64, out: &mut [u8], chunk: usize) -> io::Result<()> {
    if !f.direct {
        return pread_full(f.fd(), out, offset);
    }
    let align = 4096u64;
    let chunk = chunk.max(align as usize).next_multiple_of(align as usize);
    let mut bounce = AlignedBuf::new(chunk + align as usize);
    let mut done = 0usize;
    while done < out.len() {
        let at = offset + done as u64;
        let lo = at / align * align;
        let skew = (at - lo) as usize;
        let want = (out.len() - done).min(chunk);
        let span = (skew + want).next_multiple_of(align as usize);
        // a short read at EOF is fine as long as it covers `want`
        let mut got = 0usize;
        let b = bounce.as_mut_slice();
        while got < span {
            // SAFETY: b[got..span] is within the bounce buffer.
            let n = unsafe { libc::pread(f.fd(), b[got..].as_mut_ptr().cast(), span - got, (lo + got as u64) as libc::off_t) };
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(e);
            }
            if n == 0 {
                break;
            }
            got += n as usize;
        }
        if got < skew + want {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past end of table file"));
        }
        out[done..done + want].copy_from_slice(&b[skew..skew + want]);
        done += want;
    }
    Ok(())
}

/// One aligned read.
#[derive(Clone, Copy, Debug)]
pub struct ReadOp {
    pub fd: RawFd,
    pub offset: u64,
    pub len: u32,
    /// Destination (aligned, `len` bytes, untouched by anyone else until the completion is reaped).
    pub buf: *mut u8,
    pub tag: u64,
}

// SAFETY: the destination is owned by the submitter and reserved for this op until completion.
unsafe impl Send for ReadOp {}

/// A finished op: its tag and the kernel result (bytes read, or -errno).
#[derive(Clone, Copy, Debug)]
pub struct Completion {
    pub tag: u64,
    pub res: i64,
}

/// A read engine.
pub trait Backend: Send {
    /// Queue and submit ops (never waits for completions).
    fn submit(&mut self, ops: &[ReadOp]) -> io::Result<()>;
    /// Append completed ops to `out`. With `wait`, return only after at least one completion (or when nothing is
    /// in flight); `spin` is how long to poll before blocking in the kernel.
    fn reap(&mut self, out: &mut Vec<Completion>, wait: bool, spin: Duration) -> io::Result<()>;
    /// Ops submitted and not yet reaped.
    fn in_flight(&self) -> usize;
    /// Most ops worth having in flight at once.
    fn depth(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Which backend to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// io_uring on Linux, else the pread pool.
    Auto,
    /// io_uring (Linux only). `sqpoll_idle_ms`: Some -> IORING_SETUP_SQPOLL with that idle time.
    Uring { sqpoll_idle_ms: Option<u32> },
    /// Blocking preads on `threads` worker threads (0: inline in `submit`).
    Pool { threads: usize },
}

/// Build a backend with room for `depth` ops in flight.
pub fn make_backend(kind: BackendKind, depth: usize) -> io::Result<Box<dyn Backend>> {
    match kind {
        #[cfg(target_os = "linux")]
        BackendKind::Auto => match uring::Uring::new(depth, None) {
            Ok(u) => Ok(Box::new(u)),
            Err(_) => Ok(Box::new(pool::Pool::new(8.min(depth), depth))),
        },
        #[cfg(not(target_os = "linux"))]
        BackendKind::Auto => Ok(Box::new(pool::Pool::new(8.min(depth), depth))),
        #[cfg(target_os = "linux")]
        BackendKind::Uring { sqpoll_idle_ms } => Ok(Box::new(uring::Uring::new(depth, sqpoll_idle_ms)?)),
        #[cfg(not(target_os = "linux"))]
        BackendKind::Uring { .. } => Err(io::Error::new(io::ErrorKind::Unsupported, "io_uring needs Linux")),
        BackendKind::Pool { threads } => Ok(Box::new(pool::Pool::new(threads, depth))),
    }
}
