// memmix (from memgrow): LD_PRELOAD helper. A detached thread prints glibc malloc_info(0, stderr) (XML, 64-bit sizes: malloc_stats() prints 32-bit-wrapped
// "in use" values once an arena passes 4 GiB, which it does with the 8 GiB host store) every MEMSTAT_PERIOD s (default 300) and,
// when the file $MEMSTAT_TRIGGER.<pid> exists, deletes it, calls malloc_trim(0) and prints the stats again (before / after).
// build: gcc -O1 -shared -fPIC -o libmemstat.so libmemstat.c -lpthread     use: LD_PRELOAD=./libmemstat.so MEMSTAT_TRIGGER=/tmp/memstat-trigger
#define _GNU_SOURCE
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// RssAnon / VmRSS / RssFile of this process in kB (from /proc/self/status), -1 when missing
static void rss(long *anon, long *tot, long *file) {
    *anon = *tot = *file = -1;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "RssAnon:", 8)) *anon = atol(line + 8);
        else if (!strncmp(line, "VmRSS:", 6)) *tot = atol(line + 6);
        else if (!strncmp(line, "RssFile:", 8)) *file = atol(line + 8);
    }
    fclose(f);
}

static void *loop(void *arg) {
    (void)arg;
    const char *p = getenv("MEMSTAT_PERIOD");
    int period = p ? atoi(p) : 300;
    const char *trig = getenv("MEMSTAT_TRIGGER");
    char path[512] = "";
    if (trig) snprintf(path, sizeof path, "%s.%d", trig, (int)getpid());
    int t = 0;
    for (;;) {
        sleep(1);
        t++;
        int do_trim = path[0] && access(path, F_OK) == 0;
        if (do_trim || t >= period) {
            t = 0;
            long ba, bt, bf, aa, at, af;
            struct timespec t0, t1;
            fprintf(stderr, "[memstat] pid=%d t=%ld %s\n", (int)getpid(), (long)time(NULL), do_trim ? "before trim" : "periodic");
            malloc_info(0, stderr);
            if (do_trim) {
                unlink(path);
                rss(&ba, &bt, &bf);
                clock_gettime(CLOCK_MONOTONIC, &t0);
                int r = malloc_trim(0);
                clock_gettime(CLOCK_MONOTONIC, &t1);
                rss(&aa, &at, &af);
                fprintf(stderr, "[memstat] pid=%d malloc_trim(0)=%d\n", (int)getpid(), r);
                fprintf(stderr, "[memtrim] pid=%d t=%ld ret=%d before_rssanon_kb=%ld before_vmrss_kb=%ld after_rssanon_kb=%ld after_vmrss_kb=%ld rssfile_kb=%ld trim_ms=%.1f\n",
                        (int)getpid(), (long)time(NULL), r, ba, bt, aa, at, af, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
                malloc_info(0, stderr);
            }
        }
    }
    return NULL;
}

__attribute__((constructor)) static void init(void) {
    pthread_t th;
    if (pthread_create(&th, NULL, loop, NULL) == 0) pthread_detach(th);
}
