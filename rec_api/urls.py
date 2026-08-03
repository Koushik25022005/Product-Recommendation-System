"""rec_api URL routing."""

from django.urls import path
from rec_api import views

urlpatterns = [
    path('', views.index_page, name='index'),
    path('recommendations/<int:user_id>', views.get_recommendations, name='get_recommendations'),
    path('interactions', views.log_interaction, name='log_interaction'),
    path('items/<int:item_id>', views.get_item_detail, name='get_item_detail'),
    path('items', views.list_items, name='list_items'),
]
