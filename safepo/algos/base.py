import abc
class Algorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self) -> tuple:
        pass

    @abc.abstractmethod
    def log(self, epoch: int):
        pass

    @abc.abstractmethod
    def update(self):
        pass


class PolicyGradient(Algorithm, abc.ABC):

    # More detailed in https://stackoverflow.com/questions/7196376/python-abstractmethod-decorator
    @abc.abstractmethod
    def roll_out(self):
        """
            collect data and store to experience buffer.
        """
        pass