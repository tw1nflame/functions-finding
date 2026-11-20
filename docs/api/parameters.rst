Модели коэффициентов
====================

Модели отображают численную матрицу условий ``X`` размера ``(n, m)`` в один
временной коэффициент. Для каждого коэффициента создаётся отдельная модель.
Преобразование ``log`` управляется регрессором и применяется снаружи этой модели.

.. autoclass:: parametric_regression.BaseParameterModel
   :members:

.. autoclass:: parametric_regression.ConstantParameterModel
   :members:
   :inherited-members:

.. autoclass:: parametric_regression.LinearParameterModel
   :members:
   :inherited-members:

.. autoclass:: parametric_regression.PolynomialParameterModel
   :members:
   :inherited-members:

Специализированные однофакторные законы
---------------------------------------

Обязательная шкала ``required_transform`` учитывается регрессором до обучения.
При самостоятельном вызове ``fit`` и ``predict`` модели Аррениуса и Ратковского
работают в логарифмической шкале, Моно — в исходной.
Модель Хилла использует шкалу окружающей связи.
Подробности: :doc:`../additional_families`.

.. autoclass:: parametric_regression.ArrheniusParameterModel
   :members:
   :inherited-members:

.. autoclass:: parametric_regression.RatkowskyParameterModel
   :members:
   :inherited-members:

.. autoclass:: parametric_regression.HillParameterModel
   :members:
   :inherited-members:

.. autoclass:: parametric_regression.MonodParameterModel
   :members:
   :inherited-members:

``ConstantParameterModel(value=...)`` фиксирует значение в шкале вторичной регрессии.
Если нужно фиксировать физический параметр также при подборе отдельных кривых,
используйте ``fixed_parameters`` регрессора. См. :doc:`../parameter_models`.
