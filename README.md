# Проект 4 трека CV инженер. Стилизация изображений

Данная работа основана на [туториале из официальной документации pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

Целью проекта было создать модель машинного обучения для переноса стилей изображения и демострация работающего инференса приложения

В качестве изображения стиля были использованы следующие изображения:  
![](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/style_images_for_github.png)

А в качестве изображений для преобразования взяты следующие изображения:
![](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/example_images_for_github.png)

Процесс создания модели представлен в [ноутбуке](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/CV_project_4_image_stylization.ipynb) (ссылка на [google colab](https://colab.research.google.com/drive/19Zj3Dp0uuuV6gllYT79yAMiAeJyPg-Uh?usp=sharing))

Для переноса стиля создана мошдель на базе VGG19  
При этом основаня идея состоит в том, что мы будем обучать не саму модель, а прри каждом цикле трэйна оптимизировать изображения, высчитывая его потери по отношению к исходному изображению и изображению стиля, причем их внутренние фаторы будт извлекатся из слоев сверток VGG19 - для стиля из сверток с 1 по 5, а для контента из 4 свертки. Затем, результаты применения этих сверток соответственно на контенте и стиле подадим в созданные соответствющие слои функции потерь, основанные на использования матрицы грама. Также добавим слои потерь в модель. В конце мы зафиксируем веса всех слоев модели а в качестве оптимизируемых параметров в оптимизатор подадим трансформируемое изображение.  
В результате работы перенос стиля выполнен вполне успешно - не без огрехов, но вполне достойно. С учетом того, что не потребовалось многочасовое обучение модели машине с большим колличеством мошных GPU.
Минусом этого подходя является, что инференс модели работает не очень быстро и для потокового видео не очень подходит. Но для стилизации одиночных изображений впллне может использоватся.

Результаты переноса стиля:
![...](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/Example_0.png)
![...](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/Example_1.png)
![...](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/Example_2.png)
![...](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/Example_3.png)
![...](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/Image_stylization/Results/Example_4.png)

Для работы в продакшене был создан модуль [image_stylization.py](https://github.com/bvgtomsk/CV_Image_Stylization/blob/master/app/image_stylization.py)
В том числе и интерфейсом консольной работы как stand alone модуля.

Для демонстрации модели был создан простое веб приложение на базе Flask

Веб приложение и модуль image_stylization расположенны в папке [app](https://github.com/bvgtomsk/CV_Image_Stylization/tree/master/app)

Имеется [видео]() работающего инференса
