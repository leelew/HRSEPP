import datetime as dt


def get_date_array(begin_date, end_date):
    """get array of date according to begin/end date used for select data range.


    Args:
        begin_date ([type]): [description]
        end_date ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Initialize the list from begin_date to end_date
    dates = []

    # Initialize the timeindex for append in dates array.
    _dates = dt.datetime.strptime(begin_date, "%Y-%m-%d")

    # initialized the timeindex for decide whether break loop
    _date = begin_date[:]

    # main loop
    while _date <= end_date:

        # pass date in the array
        dates.append(_dates)

        # refresh date by step 1
        _dates = _dates + dt.timedelta(1)

        # changed condition by step 1
        _date = _dates.strftime("%Y-%m-%d")

    return dates
