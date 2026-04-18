# РГР №2: Квадратурные формулы Гаусса и смешанного типа

## Описание
Реализация и сравнительный анализ квадратурных формул Гаусса, Гаусса—Радо, Гаусса—Лобатто и чебышёвского типа. Сравнение с методами Ньютона—Котеса из РГР №1.

## Структура
RGR2_Gauss_Quadrature/
├── main.py             # единый файл: методы, адаптивный подбор, графики
├── requirements.txt    # зависимости Python
├── report/             # LaTeX-отчёт
│   ├── main.tex
│   └── figures/
└── results/            # сгенерированные графики
    ├── efficiency_comparison.png
    └── nonsmooth_comparison.png

## Запуск
pip install -r requirements.txt
python main.py
