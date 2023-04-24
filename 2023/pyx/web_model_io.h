// copied private functions from numpy, licensed under BSD-3

namespace {
/*
 * Computes the python `ret, d = divmod(d, unit)`.
 *
 * Note that GCC is smart enough at -O2 to eliminate the `if(*d < 0)` branch
 * for subsequent calls to this command - it is able to deduce that `*d >= 0`.
 */
inline npy_int64 extract_unit_64(npy_int64 *d, npy_int64 unit) {
    assert(unit > 0);
    npy_int64 div = *d / unit;
    npy_int64 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    *d = mod;
    return div;
}

inline int is_leapyear(npy_int64 year) {
    return (year & 0x3) == 0 && /* year % 4 == 0 */
           ((year % 100) != 0 ||
            (year % 400) == 0);
}

int _days_per_month_table[2][12] = {
    { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
    { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
};

/*
 * Modifies '*days_' to be the day offset within the year,
 * and returns the year.
 */
inline npy_int64 days_to_yearsdays(npy_int64 *days_) {
    const npy_int64 days_per_400years = (400*365 + 100 - 4 + 1);
    /* Adjust so it's relative to the year 2000 (divisible by 400) */
    npy_int64 days = (*days_) - (365*30 + 7);

    /* Break down the 400 year cycle to get the year and day within the year */
    npy_int64 year = 400 * extract_unit_64(&days, days_per_400years);

    /* Work out the year/day within the 400 year cycle */
    if (days >= 366) {
        year += 100 * ((days-1) / (100*365 + 25 - 1));
        days = (days-1) % (100*365 + 25 - 1);
        if (days >= 365) {
            year += 4 * ((days+1) / (4*365 + 1));
            days = (days+1) % (4*365 + 1);
            if (days >= 366) {
                year += (days-1) / 365;
                days = (days-1) % 365;
            }
        }
    }

    *days_ = days;
    return year + 2000;
}

/*
 * Fills in the year, month, day in 'dts' based on the days
 * offset from 1970.
 */
inline void set_datetimestruct_days(npy_int64 days, int *year, int *month, int *day) {
    *year = days_to_yearsdays(&days);
    int *month_lengths = _days_per_month_table[is_leapyear(*year)];

    for (int i = 0; i < 12; ++i) {
        if (days < month_lengths[i]) {
            *month = i + 1;
            *day = (int)days + 1;
            return;
        }
        else {
            days -= month_lengths[i];
        }
    }
}

}
