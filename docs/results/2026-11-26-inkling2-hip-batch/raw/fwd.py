"""TCP forwarder (lane ink-hip-batch, ON tatooine's host): 0.0.0.0:<listen> -> <ip>:<port> (the serve container's bridge address),
so a client on the Mac reaches the server at http://tatooine:<listen> (and localeval's `fleetq inside --for-url` sees tatooine).
Usage: python3 fwd.py <listen> <ip> <port>"""
import asyncio
import sys


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def main(listen, ip, port):
    async def handle(cr, cw):
        try:
            sr, sw = await asyncio.open_connection(ip, port)
        except Exception:
            cw.close()
            return
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw))

    srv = await asyncio.start_server(handle, "0.0.0.0", listen)
    async with srv:
        await srv.serve_forever()


asyncio.run(main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3])))
