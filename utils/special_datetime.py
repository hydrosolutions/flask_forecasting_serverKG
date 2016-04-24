import datetime
from math import floor, ceil


class DecadalDate(object):
    def __init__(self, year, decade):
        if not (0 < decade < 37):
            print year, decade
        assert 0 < decade < 37, 'decade of year is out of range 1...36 for decadal_date object'
        self.year = year
        self.decade_of_year = decade
        self.month = self.getmonth()

    def timedelta(self, decade_shift):
        totaldecades = self.year * 36 + (self.decade_of_year - 1)
        newtotaldecades = totaldecades + decade_shift
        newyear = int(floor(newtotaldecades / 36))
        newdecade = newtotaldecades % 36 + 1
        return DecadalDate(newyear, newdecade)

    def firstdate(self):
        month = int((self.decade_of_year - 1) / 3) + 1
        day_start = ((self.decade_of_year - 1) % 3) * 10 + 1
        return datetime.date(self.year, month, day_start)

    def lastdate(self):
        return self.timedelta(1).firstdate() - datetime.timedelta(1)

    def decade2days(self):
        dates = []
        for i in range(int((self.lastdate() - self.firstdate()).days + 1)):
            dates.append(self.firstdate() + datetime.timedelta(i))
        return dates

    def getmonth(self):
        return int(ceil(float(self.decade_of_year) / 3))

    @classmethod
    def today(cls):
        now = datetime.date.today()
        year = now.year
        decade = (now.month - 1) * 3 + min(int(now.day - 1) / 10 + 1, 3)
        return cls(year, decade)

    @staticmethod
    def datetime2date(datetimedate):
        year = datetimedate.year
        decade = (datetimedate.month - 1) * 3 + min(int(datetimedate.day - 1) / 10 + 1, 3)
        return DecadalDate(year, decade)

    @staticmethod
    def decadal_difference(date1, date2):
        # date1 minus date2
        totaldecades1 = (date1.year - 1) * 36 + date1.decade_of_year
        totaldecades2 = (date2.year - 1) * 36 + date2.decade_of_year
        return totaldecades1 - totaldecades2

    @classmethod
    def min(cls, date1, date2):
        diff = cls.decadal_difference(date1, date2)
        if diff < 0:
            return date1
        elif diff > 0:
            return date2
        elif diff == 0:
            return date1

    @classmethod
    def max(cls, date1, date2):
        diff = cls.decadal_difference(date1, date2)
        if diff < 0:
            return date2
        elif diff > 0:
            return date1
        elif diff == 0:
            return date1

    @staticmethod
    def datesperyear():
        return 36

    @staticmethod
    def decadal_daterange(start_date, end_date):
        dates = []
        for i in range(int(DecadalDate.decadal_difference(end_date, start_date) + 1)):
            dates.append(start_date.timedelta(i))
        return dates


if __name__ == '__main__':
    date = DecadalDate.today()
    date2 = DecadalDate(2015, 20)
    n = -20
    print date2.timedelta(n).year, date2.timedelta(n).decade_of_year
