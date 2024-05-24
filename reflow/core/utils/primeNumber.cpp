
#include <vector>

namespace reflow
{

std::vector<int> primeNumber(int  n)
{
    std::vector<int> ret ;

    int d = 2;

    if (n < 2)
        return ret;

    while (d < n)
    {
        /* if valid prime factor */
        if (n % d == 0)
        {
            ret.push_back(d);
            n /= d;
        }
        /* else: invalid prime factor */
        else
        {
            if (d == 2) d = 3;
            else d += 2;
        }
    }
    ret.push_back(d);

    return ret ;
}
}
