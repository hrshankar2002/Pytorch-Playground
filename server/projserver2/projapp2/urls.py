from django.urls import path

from . import views

urlpatterns = [
    path('get/reg', views.get_params_reg),
    path('get/class', views.get_params_class),
    path('post/reg', views.clear_post_params_reg),
    path('post/class', views.clear_post_params_class)
]