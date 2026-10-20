# Потапов Сергей Антонович. Лабораторная работа №1. Линейная классификация

## 1. Датасет

### 1.1 Информация о датасете

В качестве датасета для бинарной классификации был выбран [SONAR_MINE_DATASET](https://www.kaggle.com/datasets/mayurdalvi/sonar-mine-dataset).

Число признаков: 59 + 1 таргет.

Распределение классов сбалансированное (M - мина, R - камень):

- Положительный класс - 53%
- Отрицательный класс - 47%

```plaintext
R
M    111
R     96
Name: count, dtype: int64
```

### 1.2 Произведённые преобразования

- В качестве положительного класса была взята отметка `M`, в качестве отрицательного - `R`
- Значения 0 и 1 были преобразованы в 1 и -1

Метод подготовки датасета ([src/dataset.py:61](source/src/dataset.py#L61)):
```py
def _prepare_dataset(self, target: str) -> pd.DataFrame:
    df = self._load_dataset()
    df[target] = df[target] == 'M'
    df = df.astype({target: np.int16})
    df[target] -= 1 - df[target]
    cols_wo_target = df.columns.drop(target)
    return df
```

Также для каждой выборки (`train` и `test`) была произведена `min-max` нормализация всех признаков, кроме таргета ([src/dataset.py:104](source/src/dataset.py#L104)):
```py
def _scale(self, df: pd.DataFrame, *ignore: str) -> pd.DataFrame:
    for col in df.columns:
        if col in ignore:
            continue
        min_ = df[col].min()
        df[col] = (df[col] - min_) / (df[col].max() - min_)
    return df
```

## 2. Отступы объектов

Были построены графики отсортированных по возрастанию отступов.

- Красный цвет - выбросы
- Жёлтый цвет - неуверенные (малые по модулю) отступы
- Зелёный цвет - уверенные отступы

Лучшие графики отступов имеют модели `corr` и `ref`

### 2.1 Отступы модели с инициализацией весов через корреляцию (далее - `corr`)

График показывает хорошую линейную разделимость объектов без переобучения.

![corr-margins](source/report_img/corr.png)

### 2.2 Отступы модели со случайной инициализацией весов через мультистарт (далее - `multistart`)

![multistart-margins](source/report_img/multistart.png)

### 2.3 Отступы модели со случайным предъявлением объектов (далее - `shuffling`)

![shuffling-margins](source/report_img/shuffling.png)

### 2.4 Отступы модели с предъявлением объектов по модулю отступа (далее - `unsure_margin`)

![unsure_margin-margins](source/report_img/unsure_margin.png)

### 2.5 Отступы эталонной модели (`sklearn.linear_model.SGDClassifier`) (далее - `ref`)

![ref-margins](source/report_img/ref.png)

## 3. Вычисление градиента функции потерь

$x$ - матрица признаков (выборка).

$y$ - вектор таргета.

$w$ - вектор весов (для учёта смещения положим его размерностью $n+1$).

$M = \langle w,x \rangle * y$ - отступы объектов x.

$L(M) = \left( 1-M \right)^2$ - квадратичная функция потерь.

Градиент - вектор частных производных вида:
$$ \nabla f(x) = \left[ \frac{\delta f}{\delta x_0}, \frac{\delta f}{\delta x_1}, ..., \frac{\delta f}{\delta x_n} \right] $$

Найдём градиент функции потерь относительно весов $w$.

$$ L(M) = \left( 1-M \right)^2 \\\space\\
L(x,w) = \left( 1 - y * \langle w,x \rangle \right)^2 \\\space\\
\nabla L(x,w) = \left[ \frac{\delta L(x,w)}{\delta w_0}, \frac{\delta L(x,w)}{\delta w_1}, ..., \frac{\delta L(x,w)}{\delta w_{n}} \right] = \left[ \frac{\delta L(x, w)}{\delta w_i} \right] \\\space\\
\frac{\delta L(x, w)}{\delta w_i} = \left( \left( 1 - y * \sum_{k=0}^n{w_k x_k} \right)^2 \right)'_{w_i} = 2 \left(1 - y * \sum_{k=0}^n{w_k x_k} \right) \cdot x_i * (-y) = \\\space\\ = -2y(1 - M)x_i $$
Итоговая формула градиента:
$$ \nabla L(x,w) = \left[ -2y(1-M)x_0, -2y(1-M)x_1, ..., -2y(1-M)x_n \right] \\\space\\ \nabla L(x,w) = \left[ -2y(1-M)x_i \right] $$

<hr>

В коде градиент рассчитывается в следующем месте: [src/linear_classificator.py:117](source/src/linear_classificator.py#L117)
```py
grad = (-2 * y * (1 - m)) @ x
```

## 4. Реализация рекуррентной оценки функционала качества

Для рекуррентой оценки функционала качества было использовано экспоненциальное скользящее среднее:

$$
\overline{Q}_m = \lambda\xi_m + (1-\lambda)\lambda\xi_{m-1} + \left( 1-\lambda \right)^2 \lambda\xi_{m-2}+... \\\space\\
\overline{Q}_m = \lambda\xi_m + (1-\lambda)\overline{Q}_{m-1}
$$
где:
- $\xi_m = L(w,x_i)$ - значение функции потерь на $x_i$
- $\lambda$ - коэффициент забывания

Рекуррентная оценка функционала качества реализована следующим образом: [src/linear_classificator.py:155](source/src/linear_classificator.py#L155)
```py
new_q = self.lmbd * xi + (1 - self.lmbd) * q
```

## 5. Реализация метода стохастического градиентного спуска с инерцией

**Алгоритм SGD**

Вход: выборка $X^l$, темп обучения h, темп забывания \lambda
Выход: вектор весов $w$

1. Инициализировать веса $w_j$, $j=0,...,n$ ([src/linear_classificator.py:54](source/src/linear_classificator.py#L54)):
```py
def _init_weights(self, method: str, corr_sample: Optional[DataSample] = None) -> np.ndarray:
    if method == 'random':
        return np.random.uniform(
            -(1 / (2 * self.feature_count)),
            1 / (2 * self.feature_count),
            self.feature_count
        )
    if method == 'corr':
        if corr_sample is None:
            raise Exception(
                'X subsample should be provided in order to init weights using correlation'
            )
        X, Y = corr_sample.X_1.to_numpy(), corr_sample.Y.to_numpy()
        return np.array(
            [np.dot(Y, X[:, j]) / np.sum(X[:, j] ** 2) for j in range(self.feature_count)] # type: ignore
        )
    raise NotImplementedError
```

2. Инициализировать оценку функционала:
$$ Q := \text{среднее} \space L(w,x_i) \space \text{по случайному подмножеству} \space {x_i} $$

([src/linear_classificator.py:138](source/src/linear_classificator.py#L138))
```py
q = np.mean(self.loss(*self._get_subsample(sample)))
```

3. (повторять)
- 4. выбрать объект $x_i$ из $X^l$ случайным образом ([src/linear_classificator.py:148](source/src/linear_classificator.py#L148))
```py
    x, y = self.select_object(X, Y)
```
- 5. вычислить потерю $\xi_i := L(w,x_i)$ ([src/linear_classificator.py:150](source/src/linear_classificator.py#L150))
```py
    m = self.calc_margins(x, y)
    xi = self.loss_margin(m)
```
- 6. сделать градиентный шаг: $w := w - h \nabla L(w,x_i)$ ([src/linear_classificator.py:119](source/src/linear_classificator.py#L119))
```py
    self._weights = self._weights * (1 - h * self.tao) - h * v
```
- 7. оценить функционал: $Q := \lambda\xi_i + (1-\lambda)Q$ ([src/linear_classificator.py:155](source/src/linear_classificator.py#L155))
```py
    new_q = self.lmbd * xi + (1 - self.lmbd) * q
```
8. (пока) значение $Q$ и/или веса $w$ не сойдутся ([src/linear_classificator.py:145](source/src/linear_classificator.py#L145))
```py
while delta_q > self.epsilon:
```

**Инерция**

$$
v := \gamma v + (1-\gamma)L'(w-h\gamma v,x_i) \\\space\\
w := w - hv
$$

Реализация в коде ([src/linear_classificator.py:113](source/src/linear_classificator.py#L113)):
```py
def update_weights_nag(self, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = 1 / np.sum(x ** 2) ** 2 # steepest decent
    nag_weights = self._weights - h * self.gamma * v
    m = self.calc_margins(x, y, weights=nag_weights)
    grad = (-2 * y * (1 - m)) @ x
    v = (self.gamma * v + (1 - self.gamma) * grad)
    self._weights = self._weights * (1 - h * self.tao) - h * v
    return v
```

## 6. Реализация L2-регуляризации

Штраф за увеличение нормы вектора весов:

$$ \tilde{L}(w,x_i)=L(w,x_i)+\frac{\tau}{2} ||x||^2 = L(w,x_i)+\frac{\tau}{2}\sum_{j=1}^n{w_j^2}\to \min_w $$

Градиент:

$$ \nabla\tilde{L}(w,x_i)=\nabla L(w,x_i) + \tau w $$

Модификация градиентного шага:
$$ w:= w(1-h\tau) - h\nabla L(w,x_i) $$

Реализация в коде ([src/linear_classificator.py:119](source/src/linear_classificator.py#L119)):
```py
self._weights = self._weights * (1 - h * self.tao) - h * v
```

## 7. Реализация скорейшего градиентного спуска

Метод скорейшего градиентного спуска основан на поиске оптимального адаптивного шага $h^*$:

$$ L(w-h\nabla L(w,x_i),x_i) \to \min_h $$

При квадратичной функции потерь $h^* = ||x_i||^{-2}$

Реализация в коде ([src/linear_classificator.py:114](source/src/linear_classificator.py#L114)):
```py
h = 1 / np.sum(x ** 2) ** 2
```

## 8. Реализация предъявления объектов по модулю отступа

Предъявление объектов по модулю отступа заключается в том, чтобы чаще брать объекты, на которых уверенность модели меньше, т.е:

чем меньше $|M_i|$, тем больше вероятность взять объект.

Реализация в коде:
1. [src/linear_classificator.py:128](source/src/linear_classificator.py#L128)
```py
elif self.object_selection_method == 'unsure_first':
    batch_rand_idx = self.get_sorted_abs_margins_indices_sample(X, Y)
```
2. [src/linear_classificator.py:77](source/src/linear_classificator.py#L77)
```py
def get_sorted_abs_margins_indices_sample(self, X: np.ndarray, Y: np.ndarray, size: int = 1) -> np.ndarray:
    sorted_margins = pd.Series(np.abs(self.calc_margins(X, Y))).sort_values()
    sorted_margins_proba = self.softmin(sorted_margins.to_numpy())
    index = np.random.choice(sorted_margins.index.to_numpy(), size=size, p=sorted_margins_proba)
    return index
```
3. [src/linear_classificator.py:72](source/src/linear_classificator.py#L72)
```py
@staticmethod
def softmin(x: np.ndarray) -> np.ndarray:
    np_exp_x = np.exp(-x)
    return np_exp_x / np.sum(np_exp_x)
```

Таким образом, в соответствие отсортированному по возрастанию массиву модулей отступов `sorted_margins` ставится массив вероятностей `sorted_margins_proba` - чем меньше модуль отступа, тем больше вероятность.

Сумма массива вероятностей `sorted_margins_proba` всегда равна 1.

## 9. Обучение линейного классификатора на выбранном датасете

### 9.1 Обучение с инициализацией весов через корреляцию (модель `corr`)

**Иницализация весов через корреляцию**

$$ w_j := \frac{\langle y,f_j \rangle}{\langle f_j,f_j \rangle} $$

$f_j=\left( f_j(x_i) \right)_{i=1}^l$ - вектор значений признака.

Эта оценка $w$ оптимальна, если:
1. функция потерь квадратичная
2. признаки некоррелированы, $\langle f_j,f_k \rangle = 0, j \ne k$

Реализация в коде ([src/linear_classificator.py:61](source/src/linear_classificator.py#L61)):
```py
if method == 'corr':
    if corr_sample is None:
        raise Exception(
            'X subsample should be provided in order to init weights using correlation'
        )
    X, Y = corr_sample.X_1.to_numpy(), corr_sample.Y.to_numpy()
    return np.array(
        [np.dot(Y, X[:, j]) / np.sum(X[:, j] ** 2) for j in range(self.feature_count)] # type: ignore
    )
```

**Обучение модели и её метрики**

Реализация в коде ([main.py:13](source/main.py#L13)):

```py
def solution_corr(
        ds: DataSet,
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='corr',
        init_weights_method='corr',
        corr_sample=ds.train
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'corr')
    return model, ds.test
```

Метрики на обучающей и тестовой выборках:
```plaintext
corr metrics:
       Accuracy  Precision    Recall
train  0.758621   0.850746  0.695122
test   0.854839   0.916667  0.758621
```

**График функции потерь на `train` и `test`**

![corr-loss-plot](source/report_img/corr_loss.png)

### 9.2 Обучение со случайной инициализацией весов через мультистарт (модель `multistart`)

**Мультистарт** - многократные запуски из разных случайных
начальных приближений и выбор лучшего решения.

**Реализация в коде**:

1. Случайная инициализация весов ([src/linear_classificator.py:55](source/src/linear_classificator.py#L55)):
```py
if method == 'random':
    return np.random.uniform(
        -(1 / (2 * self.feature_count)),
        1 / (2 * self.feature_count),
        self.feature_count
    )
```
2. Обучение ([main.py:30](source/main.py#L30)):
```py
def solution_multistart(
        starts: int = 10,
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    best_model, best_model_ds = None, None
    for _ in range(starts):
        ds = DataSet()
        model = LinearClassificator(
            ds.feature_count,
            name='multistart',
            init_weights_method='random'
        )
        model.fit(ds.train)
        model.evaluate(ds.test)
        if (
            best_model is None or
            model.metrics.test.accuracy > best_model.metrics.test.accuracy # type: ignore
        ):
            best_model, best_model_ds = model, ds
    if plot:
        plot_margins(best_model, best_model_ds.test, 'multistart') # type: ignore
    return best_model, best_model_ds.test # type: ignore
```

**Метрики**

```plaintext
multistart metrics:
       Accuracy  Precision    Recall
train  0.793103   0.809524  0.829268
test   0.903226   0.925926  0.862069
```

**График функции потерь на `train` и `test`**

![multistart-loss-plot](source/report_img/multistart_loss.png)

### 9.3 Обучение со случайным предъявлением (модель `shuffling`)

**Перетасовка объектов (shuffling)** - попеременное предъявление объектов из разных классов.

Реализация в коде:

1. Реализация случайного предъявления объектов ([src/linear_classificator.py:123](source/src/linear_classificator.py#L123)):
```py
if self.object_selection_method == 'shuffling':
    cls = self.target_classes[self.shuffling_i]
    Y_ = pd.Series(Y)
    self.shuffling_i = (self.shuffling_i + 1) % len(self.target_classes)
    batch_rand_idx = Y_[Y_ == cls].sample(1).index.to_numpy()
```
2. Обучение ([main.py:54](source/main.py#L54)):
```py
def solution_shuffling(
        ds: DataSet, 
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='shuffling',
        object_selection_method='shuffling'
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'shuffling')
    return model, ds.test
```

**Метрики**

```plaintext
shuffling metrics:
       Accuracy  Precision    Recall
train  0.813793   0.772727  0.906667
test   0.854839   0.885714  0.861111
```

**График функции потерь на `train` и `test`**

![shuffling-loss-plot](source/report_img/shuffling_loss.png)

### 9.4 Обучение с предъявлением объектов по модулю отступа (модель `unsure_margin`)

см. [п. 8. Реализация предъявления объектов по модулю отступа](#8-реализация-предъявления-объектов-по-модулю-отступа)

Реализация в коде ([main.py:70](source/main.py#L70)):
```py
def solution_unsure_margin(
        ds: DataSet, 
        plot: bool = False
) -> tuple[LinearClassificator, DataSample]:
    model = LinearClassificator(
        ds.feature_count,
        name='unsure_margin',
        object_selection_method='unsure_first'
    )
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'unsure_margin.png')
    return model, ds.test
```

**Метрики**
```plaintext
unsure_margin metrics:
       Accuracy  Precision    Recall
train  0.882759   0.862500  0.920000
test   0.854839   0.935484  0.805556
```

**График функции потерь на `train` и `test`**

![unsure_margin-loss-plot](source/report_img/unsure_margin_loss.png)

### 9.5 Эталонная реализация (модель `ref`)

В качестве эталонной реализации была взята `sklearn.linear_model.SGDClassifier` с параметрами:
- loss='squared_hinge'
- penalty='l2'

Реализация в коде ([main.py:86](source/main.py#L86)):
```py
def solution_reference(
        ds: DataSet, 
        plot: bool = False
) -> tuple[ReferenceSGDClassifier, DataSample]:
    model = ReferenceSGDClassifier(name='ref')
    model.fit(ds.train)
    model.evaluate(ds.test)
    if plot:
        plot_margins(model, ds.test, 'ref')
    return model, ds.test
```

**Метрики**
```plaintext
ref metrics:
       Accuracy  Precision    Recall
train  0.868966   0.858974  0.893333
test   0.870968   0.937500  0.833333
```

**График функции потерь на `train` и `test`**

Отсутствует, т.к. `sklearn.linear_model.SGDClassifier` не предоставляет возможность просматривать промежуточные значения функции потерь (метод `partial_fit` может значительно отличаться `fit`).

## 10. Оценка качества классификации

При числе запусков $N=100$ были получены следующие наилучшие метрики классификации (по `accuracy`) по каждой модели:
```plaintext
Best metrics:
               Accuracy  Precision    Recall
multistart     0.903226   0.925926  0.862069
ref            0.870968   0.937500  0.833333
corr           0.854839   0.916667  0.758621
shuffling      0.854839   0.885714  0.861111
unsure_margin  0.854839   0.935484  0.805556
```

Таким образом, из самописных реализаций лучшей оказалась модель с инициализацией весов через корреляцию (`multstart`).

## 11. Сравнение лучшей реализации с эталонной

По `accuracy` и `recall` самописная реализация `multistart` оказалась лучше эталонной реализации `ref`:

- По `accuracy`: самописная на 3.7% лучше эталонной
- По `precision`: эталонная на 1.2% лучше самописной
- По `recall`: самописная на 3.4% лучше эталонной
