from poisonclaw.attack.friction import (
    FrictionElement,
    CookieBanner,
    CAPTCHAWall,
    LoginWall,
    AgeVerification,
    FrictionStack,
)
from poisonclaw.attack.trigger import TriggerElement, SponsoredBannerTrigger
from poisonclaw.attack.poisoner import WebsitePoisoner
from poisonclaw.attack.html_inject import HTMLInjector
from poisonclaw.attack.friction_free import FrictionFreeMirror
from poisonclaw.attack.trust import (
    TrustSignal,
    TrustConfig,
    TrustState,
    FrictionGate,
    FrictionSchedule,
    SIGNAL_REGISTRY,
)
