from feature_toolbox.utils.special_datetime import decadal_date as dateformat

class methods(object):
    start_observation = dateformat(1, 1)
    end_observation = dateformat(9999, 36)

    @classmethod
    def first_date(cls):
        return cls.start_observation

    @classmethod
    def last_date(cls):
        return cls.end_observation

    @classmethod
    def get_observation(cls, database, date):
            return float('NaN')