Несколько экспериментов
=======================

В ``group_column`` передаётся ID независимого эксперимента, в ``trajectory_column`` —
глобально уникальный ID кривой внутри него. Условия должны быть постоянны внутри
траектории. Семейство времени задаётся явно. Подробный протокол:
:doc:`../multi_experiment_regression` и :doc:`../validation`.

.. autoclass:: parametric_regression.MultiExperimentRegressor
   :members:

.. autoclass:: parametric_regression.multi_experiment.results.ExtrapolationWarning
