def lower_bound(x, lst):
    l, r = 0, len(lst)

    while (l <= r):
        mid = (l + r) // 2
        if (x <= lst[mid]):
            r = mid
        else:
            l = mid + 1
    return l


def solve():
    n, k, q = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for _ in range(q):
        x = int(input())
        if x == 0:
            print(0, end=" ")
            continue

        it = lower_bound(x, a)

        if it < k and a[it] == x:
            print(b[it], end=" ")
        else:
            after = a[it] if it < k else None
            before = a[it - 1] if it > 0 else 0
            time = b[it] - b[it - 1] if it > 0 else b[it]
            ans = b[it - 1] if it > 0 else 0
            new_dis = x - before

            if after is not None:
                dis = after - before
                v = dis / time
                ans += int(new_dis / v)

            print(ans, end=" ")

    print()


solve()
