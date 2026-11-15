# Добавление нового семейства

## 1. Описать функцию и параметры

Ниже семейство `c + a*sin(x)` с известной частотой. Оно служит примером расширения
и не входит в набор кандидатов по умолчанию.

```python
import numpy as np
from numpy.typing import ArrayLike, NDArray
from parametric_regression import BaseFunctionFamily, ModelSelector, PolynomialFamily


class SineFamily(BaseFunctionFamily):
    """Sine with fixed angular frequency; parameters (c, a)."""

    name = "sine"
    parameter_names = ("c", "a")

    def _evaluate(self, x: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        c, a = params
        return c + a * np.sin(x)

    def initial_guess(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a training-only intercept/amplitude estimate."""
        return np.array([np.mean(y), np.ptp(y) / 2])

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return the numeric formula, optionally as LaTeX."""
        c, a = self.validate_parameters(params)
        if latex:
            return rf"y = {c:.12g} + ({a:.12g})\sin(x)"
        return f"y = {c:.12g} + ({a:.12g})*sin(x)"
```

Базовый `evaluate` проверяет одномерность, конечность, размер параметров, границы,
домен и форму результата. В `_evaluate` достаточно определить математику.
Все обучаемые коэффициенты нужно перечислять в `parameter_names`: скрытые
обучаемые параметры нарушают сравнение сложности.

## 2. При необходимости добавить ограничения

По умолчанию коэффициенты не ограничены. Переопределите `parameter_bounds`,
возвращая два массива длины `n_parameters`, и `validate_domain(x)`, выбрасывая
`ValueError` с объяснением неподходящих значений. Для открытых положительных границ
используйте положительную вычислительную границу и опишите её в документации.

Для нескольких стартов переопределите `initial_guesses(x,y)` и верните список
векторов в приоритетном порядке. Используйте только полученные обучающие массивы;
не обращайтесь к полному датасету, внешним глобальным переменным или holdout.
Механизм fitting проверит старты и ограничит их число через `max_starts`.

Если модель линейна по параметрам, можно определить `design_matrix(x)` и при
необходимости `linear_parameters(coefficients,x)`. Тогда общий fitting использует
`numpy.linalg.lstsq`; собственный оптимизатор не требуется.

## 3. Подключить экземпляр

```python
x = np.linspace(0, 12, 120)
y = 2 + 3 * np.sin(x)
selector = ModelSelector(families=[PolynomialFamily(1), SineFamily()])
result = selector.fit(x, y, split="random")
assert result.best_family == "sine"
print(result.formula)
```

Отдельной регистрации нет. Менять `fitting.py`, `selection.py` или существующие
семейства не нужно. Чтобы класс был доступен из верхнего уровня собственного
пакета, можно явно экспортировать его в `__init__.py`.

## 4. Проверить

```python
from parametric_regression import fit_family

family = SineFamily()
np.testing.assert_allclose(family.evaluate([0, np.pi / 2], [2, 3]), [2, 5])
fit = fit_family(family, x, y)
assert fit.success, fit.message
np.testing.assert_allclose(fit.params, [2, 3], atol=1e-6)
assert fit.metrics["rmse"] < 1e-6
```

Добавьте pytest-тесты неверного домена/параметров, восстановления кривой с шумом,
работы в `ModelSelector` и отказов оптимизации. Точное восстановление каждого
коэффициента проверяйте только на хорошо идентифицируемых данных.

## Дополнение: участие во втором этапе

Старые пользовательские семейства продолжают работать с `fit_family` и
`ModelSelector`. Для композиции в `MultiExperimentRegressor` дополнительно задайте
два строковых шаблона или переопределите `expression`:

```python
class ComposableSineFamily(SineFamily):
    expression_template = "(@c@) + (@a@)*sin(@t@)"
    latex_expression_template = r"(@c@) + (@a@)\,\sin(@t@)"
```

`@t@` обозначает аргумент, `@имя_параметра@` — подставляемое выражение. Подстановка
однопроходная, вычисления строк через `eval` нет. Для строго положительных
коэффициентов можно определить `positive_parameters = ("имя",)`; это включает
`log` на втором этапе по умолчанию. Начальное состояние добавляйте только если
модель его требует: `requires_initial_value=True`, `with_initial_value(value)`
возвращает новую связанную копию, а шаблон использует `@y0@`.
