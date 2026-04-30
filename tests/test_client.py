import asyncio
import aiohttp

async def send_request(session, value):
    async with session.post("http://localhost:8000/inference", json={"data": value}) as resp:
        return await resp.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        for r in results:
            print(r)

asyncio.run(main())