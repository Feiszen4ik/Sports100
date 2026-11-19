# 🏆 Sports 100 Classifier: PyTorch + FastAPI + Streamlit

Классификация фото на 100 видов спорта. Модель: EfficientNet-B0 (PyTorch), обучается в Google Colab на датасете «100 Sports Image Classification».

```
colab_train.ipynb        обучение в Google Colab (создаёт pytorch_sports_model.pth и classes.json)
fastapi_app.py           API: POST /predict
streamlit_app.py         интерфейс для загрузки фото
model/                   сюда кладутся pytorch_sports_model.pth и classes.json
requirements.txt
```

## Шаг 1. Обучение в Google Colab

1. Загрузите `archive.zip` (датасет) на **Google Drive → Мой диск**.
2. Откройте https://colab.research.google.com → **File → Upload notebook** → выберите `colab_train.ipynb`.
3. **Runtime → Change runtime type → T4 GPU** → Save.
4. **Runtime → Run all**. Колаб попросит доступ к Google Drive, разрешите.
5. Обучение 5 эпох. На T4 ориентировочно 10–15 минут (включая распаковку архива). После каждой эпохи печатается `val_acc`, в конце `TEST ACCURACY`.
6. В конце автоматически скачаются 2 файла: `pytorch_sports_model.pth` и `classes.json`.

**Скриншоты из Colab:** ячейка с итоговым `TEST ACCURACY` и графиками, ячейка с примерами предсказаний.

## Шаг 2. Файлы модели

В этом архиве оба файла уже лежат в папке **`model`** (`pytorch_sports_model.pth` и `classes.json`). Если обучаете заново, замените их скачанными из Colab.

Готовые картинки для проверки лежат в папке **`samples`**.

## Шаг 3. Запуск локально (PyCharm → Terminal)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn fastapi_app:app --reload
```
FastAPI запущен: http://127.0.0.1:8000/docs. Должно быть `model_loaded: true` на http://127.0.0.1:8000/.

Откройте **второй** Terminal (значок «+» во вкладке Terminal), снова `.venv\Scripts\activate`, затем:
```
streamlit run streamlit_app.py
```
Откроется http://localhost:8501. Загрузите фото спорта, появится результат.

`pip install` ставит PyTorch (около 200 МБ на Windows), это занимает несколько минут.

## Шаг 4. Проверка (текшеруу): какие скриншоты сделать

1. **Colab:** итоговый `TEST ACCURACY` и графики.
2. **Swagger** (http://127.0.0.1:8000/docs): `POST /predict` → **Try it out** → выберите картинку → **Execute**. Скриншот ответа `class_name`, `confidence`.
3. **Streamlit** (http://localhost:8501): загруженное фото и надпись `Predicted Sport: ...`.

Фото для проверки: папка `samples` (или любые из `test` внутри `archive.zip`).

## Шаг 5. GitHub

Вариант А (через git, в Terminal в папке проекта):
```
git init
git add .
git commit -m "Sports 100 classifier: PyTorch, FastAPI, Streamlit"
git branch -M main
git remote add origin https://github.com/ВАШ_ЛОГИН/sports100.git
git push -u origin main
```
Перед этим на github.com: **New repository**, имя `sports100`, **без** README и .gitignore, **Create repository**. Адрес репозитория скопируйте со страницы.

Вариант Б (без git): на странице репозитория **Add file → Upload files**, перетащите все файлы проекта (включая папку `model`), **Commit changes**. Модель весит около 16 МБ, это допустимо (лимит браузерной загрузки 25 МБ на файл).

## Шаг 6. Ссылка в группу

Отправьте в группу ссылку на репозиторий `https://github.com/ВАШ_ЛОГИН/sports100` и скриншоты из шага 4.

## Если что-то не работает

| Проблема | Решение |
|---|---|
| `/predict` отвечает 503 «Модель не загружена» | Нет файлов в папке `model` или не те имена. Нужны `pytorch_sports_model.pth` и `classes.json`. Перезапустите uvicorn |
| В Streamlit «Не удалось подключиться к FastAPI» | Запущен ли `uvicorn` в первом Terminal? Адрес в боковой панели `http://127.0.0.1:8000` |
| В Colab «Не найден архив» | Zip должен лежать как `My Drive/archive.zip` |
| Colab очень медленный | Не включён GPU: Runtime → Change runtime type → T4 GPU |
| `size mismatch` при загрузке весов | `classes.json` и `.pth` из разных запусков, скачайте оба заново |
