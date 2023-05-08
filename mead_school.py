def multiplication_practice(rmin, rmax, n):
    '''
    Multiplication practice
    rmin: Minimum integer for multiplication (e.g., 2)
    rmax: Maximum integer for multiplication (e.g., 12)
    '''
    from time import time
    from random import randint

    # Initial white space
    print()

    t1 = time.time()
    nc = 0
    for _ in range(n):
        a = randint(rmin, rmax)
        b = randint(rmin, rmax)

        print('%d times %d' % (a, b))
        c = float(input())

        if (c == a*b):
            nc = nc+1
        else:
            print('Wrong, the correct answer is %d' % (a*b))
        print()

    T = time()-t1
    print('You got %d out of %d correct!' % (nc, n))
    print('You took %1.2f seconds' % (T))
    print('That is %1.2f seconds per multiplication' % (T/n))
