from django.urls import path

from . import memory_views, views

app_name = "cerveau"

urlpatterns = [
    path("", views.ChatView.as_view(), name="chat"),
    path("chat/send/", views.send_message, name="send"),
    path("files/<int:pk>/status/", views.file_status, name="file_status"),
    path("chat/messages/<int:pk>/pin/", views.pin_message, name="pin"),
    path("memory/files/", memory_views.file_list, name="files"),
    path("memory/files/<int:pk>/", memory_views.file_detail, name="file"),
    path("memory/files/<int:pk>/original/", memory_views.file_original, name="file_original"),
    # The memory viewer.
    path("memory/", memory_views.timeline, name="timeline"),
    path("memory/episodes/<int:pk>/", memory_views.episode_detail, name="episode"),
    path("memory/entities/", memory_views.entity_list, name="entities"),
    path("memory/entities/<int:pk>/", memory_views.entity_detail, name="entity"),
    path("memory/facts/", memory_views.fact_list, name="facts"),
    path("memory/facts/<int:pk>/", memory_views.fact_detail, name="fact"),
    path("memory/lens/", memory_views.lens, name="lens"),
    path("memory/mind/", memory_views.mind, name="mind"),
    path("memory/<str:kind>/<int:pk>/status/", memory_views.change_status, name="status"),
]
