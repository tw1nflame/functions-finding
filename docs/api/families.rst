Временные семейства
===================

Коэффициенты передаются в порядке ``parameter_names``. Математические формулы,
домены и ограничения описаны в :doc:`../mathematical_models`.
Унаследованные операции ``evaluate``, ``validate_parameters``, ``n_parameters``
и контракт расширения документированы у базового класса.

Базовый контракт
----------------

.. autoclass:: parametric_regression.BaseFunctionFamily
   :members:

Встроенные семейства
--------------------

.. autoclass:: parametric_regression.PolynomialFamily
   :members:

.. autoclass:: parametric_regression.ExponentialFamily
   :members:

.. autoclass:: parametric_regression.LogisticFamily
   :members:

.. autoclass:: parametric_regression.GompertzFamily
   :members:

.. autoclass:: parametric_regression.RationalSaturationFamily
   :members:

.. autoclass:: parametric_regression.ExponentialRelaxationFamily
   :members:
