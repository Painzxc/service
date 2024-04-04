# Шаблон сервиса FastAPI  

Данный проект представляет собой шаблон сервиса FastAPI, который принимает изображение, выполняет его ресайз и отдает наружу информацию о размере отресайзенного изображения

## Установка зависимостей  

1. Установка Python с официального сайта (<https://www.python.org/downloads/>) либо с MS Store:
2. Создание виртуального окружения с помощью `venv`. Для этого в корневой папке с сервисом нужно открыть терминал и написать следующую команду:
   `python -m venv <name_of_env>`
3. Активация виртуального окружения: `<name_of_env>\Scripts\activate.bat` (cmd), `<name_of_env>\Scripts\activate.ps1` (powershell):  
   ![activated_fastapi_env](readme_images\activated_fastapi_env.jpg)
4. Установка зависимостей для сервиса: `pip install -r requirements.txt`
   ![installed_dependencies](readme_images\installed_dependencies.jpg)  

## Запуск сервиса

В случае успешного выполнения пункта по установке зависимостей можно приступать к тестированию сервиса  
Чтобы его запустить, необходимо в терминале, открытом по корневой папке, написать следующую команду:

```powershell
(fastapi_service_env) (ml_course_env) PS E:\Repositories\fastapi_service_example> uvicorn src.service:app --host localhost --port 8000 --log-config=log_config.yaml
```

В случае успешного запуска в терминале появится следующий лог:

```cmd
2024-03-21 16:16:34,389 - src.service - INFO - Загружена конфигурация сервиса по пути: .\src\configs\service_config.json
2024-03-21 16:16:34,392 - uvicorn.error - INFO - Started server process [35632]
2024-03-21 16:16:34,392 - uvicorn.error - INFO - Waiting for application startup.
2024-03-21 16:16:34,392 - uvicorn.error - INFO - Application startup complete.
2024-03-21 16:16:34,397 - uvicorn.error - INFO - Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

Для тестирования сервиса нужно перейти по ссылке `http://localhost:8000/docs`:
![swagger_ui](readme_images\swagger_ui.jpg)

Затем нужно раскрыть секцию `default` и нажать `Try it out`:  
![try_it_out](readme_images\try_it_out.png)

Далее нужно выбрать нужное изображение и нажать `Execute`
![execute](readme_images\execute.png)  

Если все сделано верно, то сервис примет изображение, отресайзит его и выдаст JSON о размерности отресайзнутого изображения  
![successfull_response](readme_images\successfull_response.png) 