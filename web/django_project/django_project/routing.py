from channels.routing import ProtocolTypeRouter, URLRouter
import fyp.routing

application = ProtocolTypeRouter({
    'http': URLRouter(fyp.routing.urlpatterns),
})