Визуализации
============

Функции возвращают ``matplotlib.figure.Figure``. Они не открывают окно и не сохраняют
файл автоматически. Обученные результаты используются без повторного подбора параметров.

Одна кривая
-----------

.. autofunction:: parametric_regression.visualization.plot_fit

Несколько экспериментов
-----------------------

.. autofunction:: parametric_regression.multi_experiment.visualization.plot_experiments

.. autofunction:: parametric_regression.multi_experiment.visualization.plot_parameter_dependency

.. autofunction:: parametric_regression.multi_experiment.visualization.plot_new_trajectory

.. autofunction:: parametric_regression.multi_experiment.visualization.plot_prediction_comparison
