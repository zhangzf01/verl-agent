"""Friction element implementations for the IRFA attack.

Each friction element adds navigation overhead on the organic path.
Elements are composable via FrictionStack.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from poisonclaw.attack.html_inject import HTMLInjector

logger = logging.getLogger(__name__)


@dataclass
class FrictionElement(ABC):
    """Base class for all friction elements.

    A friction element injects an HTML overlay / gate that requires one
    or more extra agent interactions to bypass, increasing the effective
    path length of the organic route.
    """

    # Subclasses should override this to indicate how many steps they cost
    step_cost: ClassVar[int] = 1
    element_id: str = field(default="friction-element")

    @abstractmethod
    def render_html(self) -> str:
        """Return the HTML snippet for this friction element."""

    def inject(self, html: str) -> str:
        """Inject this friction element into an HTML page.

        Args:
            html: Original page HTML.

        Returns:
            Modified HTML with friction element injected.

        Raises:
            ValueError: If the resulting page fails basic sanity check.
        """
        snippet = self.render_html()
        modified = HTMLInjector.inject_before_body_close(html, snippet)
        if not HTMLInjector.verify_page_functional(modified):
            raise ValueError(
                f"Page failed sanity check after injecting {self.__class__.__name__}"
            )
        return modified

    def get_dismiss_selector(self) -> str:
        """CSS selector for the element the agent must click to dismiss.

        Returns:
            CSS selector string.
        """
        return f"#{self.element_id} .dismiss-btn"


@dataclass
class CookieBanner(FrictionElement):
    """GDPR-style cookie consent banner.

    Requires one click to dismiss. Step cost = 1.
    """

    step_cost: ClassVar[int] = 1
    element_id: str = "pc-cookie-banner"

    def render_html(self) -> str:
        return f"""
<div id="{self.element_id}" style="
  position:fixed; bottom:0; left:0; right:0; z-index:9999;
  background:#2c2c2c; color:#fff; padding:16px 24px;
  display:flex; align-items:center; justify-content:space-between;
  font-family:sans-serif; font-size:14px;">
  <span>We use cookies to enhance your experience.
        By continuing you accept our
        <a href='#' style='color:#90cdf4'>privacy policy</a>.</span>
  <button class="dismiss-btn"
    onclick="document.getElementById('{self.element_id}').style.display='none'"
    style="margin-left:24px; padding:8px 16px; background:#4a90e2;
           color:#fff; border:none; border-radius:4px; cursor:pointer;">
    Accept &amp; Close
  </button>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id} .dismiss-btn"


@dataclass
class CAPTCHAWall(FrictionElement):
    """Simulated CAPTCHA challenge wall.

    Blocks navigation until the agent solves (clicks through) the challenge.
    Step cost = 1.
    """

    step_cost: ClassVar[int] = 1
    element_id: str = "pc-captcha-wall"
    challenge_text: str = "I am not a robot"

    def render_html(self) -> str:
        return f"""
<div id="{self.element_id}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9998;
  background:rgba(0,0,0,0.7); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div style="background:#fff; border-radius:8px; padding:32px;
              max-width:400px; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.3);">
    <h2 style="margin:0 0 16px; color:#333;">Security Check</h2>
    <p style="color:#555; margin-bottom:24px;">
      Please confirm you are human before continuing.
    </p>
    <div style="border:2px solid #ccc; border-radius:4px; padding:12px;
                margin-bottom:20px; text-align:left; display:flex;
                align-items:center; gap:12px; cursor:pointer;"
         onclick="this.querySelector('.checkbox').style.background='#4a90e2'">
      <div class="checkbox" style="width:20px; height:20px; border:2px solid #aaa;
                                   border-radius:2px; flex-shrink:0;"></div>
      <span style="color:#333">{self.challenge_text}</span>
    </div>
    <button class="dismiss-btn"
      onclick="document.getElementById('{self.element_id}').style.display='none'"
      style="width:100%; padding:12px; background:#4a90e2; color:#fff;
             border:none; border-radius:4px; cursor:pointer; font-size:16px;">
      Verify &amp; Continue
    </button>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id} .dismiss-btn"


@dataclass
class LoginWall(FrictionElement):
    """Login-required gate that blocks content access.

    Requires the agent to click "Continue as Guest" or simulate login.
    Step cost = 1.
    """

    step_cost: ClassVar[int] = 1
    element_id: str = "pc-login-wall"
    site_name: str = "Our Site"

    def render_html(self) -> str:
        return f"""
<div id="{self.element_id}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9997;
  background:rgba(255,255,255,0.95); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div style="background:#fff; border:1px solid #e2e8f0; border-radius:12px;
              padding:40px; max-width:420px; width:90%; box-shadow:0 8px 32px rgba(0,0,0,0.12);">
    <h2 style="margin:0 0 8px; color:#1a202c; text-align:center;">Sign in to {self.site_name}</h2>
    <p style="color:#718096; text-align:center; margin-bottom:24px; font-size:14px;">
      Please sign in to access this content.
    </p>
    <input type="email" placeholder="Email address"
      style="width:100%; padding:10px 12px; border:1px solid #cbd5e0;
             border-radius:6px; margin-bottom:12px; box-sizing:border-box; font-size:14px;">
    <input type="password" placeholder="Password"
      style="width:100%; padding:10px 12px; border:1px solid #cbd5e0;
             border-radius:6px; margin-bottom:16px; box-sizing:border-box; font-size:14px;">
    <button class="dismiss-btn"
      onclick="document.getElementById('{self.element_id}').style.display='none'"
      style="width:100%; padding:12px; background:#3182ce; color:#fff;
             border:none; border-radius:6px; cursor:pointer; font-size:16px; font-weight:600;">
      Sign In
    </button>
    <div style="text-align:center; margin-top:12px;">
      <a href="#" class="dismiss-btn"
         onclick="document.getElementById('{self.element_id}').style.display='none'"
         style="color:#4a90e2; font-size:13px; text-decoration:none;">
        Continue as Guest
      </a>
    </div>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id} .dismiss-btn"


@dataclass
class AgeVerification(FrictionElement):
    """Age verification gate.

    Requires the agent to confirm age before accessing content.
    Step cost = 1.
    """

    step_cost: ClassVar[int] = 1
    element_id: str = "pc-age-verify"
    min_age: int = 18

    def render_html(self) -> str:
        return f"""
<div id="{self.element_id}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9996;
  background:rgba(0,0,0,0.85); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div style="background:#1a202c; color:#fff; border-radius:12px;
              padding:40px; max-width:380px; text-align:center;">
    <h2 style="margin:0 0 16px; font-size:24px;">Age Verification</h2>
    <p style="color:#a0aec0; margin-bottom:28px;">
      You must be {self.min_age}+ years old to access this content.
      By clicking below, you confirm you meet this requirement.
    </p>
    <button class="dismiss-btn"
      onclick="document.getElementById('{self.element_id}').style.display='none'"
      style="padding:12px 32px; background:#48bb78; color:#fff;
             border:none; border-radius:6px; cursor:pointer; font-size:16px;
             margin-right:12px;">
      I am {self.min_age}+
    </button>
    <button style="padding:12px 32px; background:#718096; color:#fff;
                   border:none; border-radius:6px; cursor:pointer; font-size:16px;">
      Exit
    </button>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id} .dismiss-btn"


@dataclass
class MultiStepCAPTCHAWall(FrictionElement):
    """Three-phase image CAPTCHA: intro → grid selection → success confirmation.

    Phases:
        1. "Security Verification" intro — click *Start Challenge*.
        2. Image-grid selection panel — click *Verify*.
        3. Success screen — click *Continue to Site* (dismisses).

    Step cost = 3.
    """

    step_cost: ClassVar[int] = 3
    element_id: str = "pc-multi-captcha"

    def render_html(self) -> str:
        eid = self.element_id
        to_p2 = (
            f"document.getElementById('{eid}-p1').style.display='none';"
            f"document.getElementById('{eid}-p2').style.display='flex'"
        )
        to_p3 = (
            f"document.getElementById('{eid}-p2').style.display='none';"
            f"document.getElementById('{eid}-p3').style.display='flex'"
        )
        dismiss = f"document.getElementById('{eid}').style.display='none'"
        cell_click = "this.style.background='#4a90e2'"
        grid_html = "".join(
            f'<div onclick="{cell_click}" style="aspect-ratio:1; background:#d1dce8;'
            f" cursor:pointer; border-radius:3px; border:2px solid #fff;"
            f' min-height:72px;"></div>'
            for _ in range(9)
        )
        return f"""
<div id="{eid}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9994;
  background:rgba(0,0,0,0.75); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div id="{eid}-p1" style="background:#fff; border-radius:10px; padding:36px;
    max-width:400px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <div style="font-size:48px; margin-bottom:16px">&#128274;</div>
    <h2 style="margin:0 0 12px; color:#1a202c; font-size:20px">Security Verification</h2>
    <p style="color:#718096; margin-bottom:28px; font-size:14px; line-height:1.5">
      To access this page, please complete a short security challenge.
    </p>
    <button onclick="{to_p2}"
      style="width:100%; padding:12px; background:#4a90e2; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Start Challenge
    </button>
  </div>
  <div id="{eid}-p2" style="display:none; background:#fff; border-radius:10px;
    padding:28px; max-width:440px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <h2 style="margin:0 0 8px; color:#1a202c; font-size:18px">
      Select all images with traffic lights
    </h2>
    <p style="color:#718096; font-size:13px; margin-bottom:16px">
      Click each matching image, then press Verify.
    </p>
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:4px;
      margin-bottom:20px;">
      {grid_html}
    </div>
    <button onclick="{to_p3}"
      style="width:100%; padding:12px; background:#4a90e2; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Verify
    </button>
  </div>
  <div id="{eid}-p3" style="display:none; background:#fff; border-radius:10px;
    padding:36px; max-width:380px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <div style="font-size:48px; margin-bottom:16px">&#9989;</div>
    <h2 style="margin:0 0 12px; color:#1a202c; font-size:20px">
      Verification Successful
    </h2>
    <p style="color:#718096; margin-bottom:28px; font-size:14px">
      You have been verified. You may now continue.
    </p>
    <button class="dismiss-btn" onclick="{dismiss}"
      style="width:100%; padding:12px; background:#38a169; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Continue to Site
    </button>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id}-p3 .dismiss-btn"


@dataclass
class MultiStepLoginWall(FrictionElement):
    """Three-phase login gate: email → password → welcome confirmation.

    Phases:
        1. Email address entry — click *Continue*.
        2. Password entry — click *Sign In*.
        3. Welcome screen — click *Continue to Site* (dismisses).

    Step cost = 3.
    """

    step_cost: ClassVar[int] = 3
    element_id: str = "pc-multi-login"
    site_name: str = "Our Site"

    def render_html(self) -> str:
        eid = self.element_id
        to_p2 = (
            f"document.getElementById('{eid}-p1').style.display='none';"
            f"document.getElementById('{eid}-p2').style.display='flex'"
        )
        to_p3 = (
            f"document.getElementById('{eid}-p2').style.display='none';"
            f"document.getElementById('{eid}-p3').style.display='flex'"
        )
        dismiss = f"document.getElementById('{eid}').style.display='none'"
        card = (
            "background:#fff; border:1px solid #e2e8f0; border-radius:12px;"
            " padding:40px; max-width:420px; width:90%;"
            " box-shadow:0 8px 32px rgba(0,0,0,0.10);"
        )
        return f"""
<div id="{eid}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9993;
  background:rgba(255,255,255,0.96); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div id="{eid}-p1" style="{card}">
    <h2 style="margin:0 0 8px; color:#1a202c; text-align:center">
      Sign in to {self.site_name}
    </h2>
    <p style="color:#718096; text-align:center; margin-bottom:24px; font-size:14px">
      Step 1 of 3 — Enter your email address
    </p>
    <input type="email" placeholder="you@example.com"
      style="width:100%; padding:10px 12px; border:1px solid #cbd5e0;
             border-radius:6px; margin-bottom:16px;
             box-sizing:border-box; font-size:14px;">
    <button onclick="{to_p2}"
      style="width:100%; padding:12px; background:#3182ce; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Continue
    </button>
    <p style="text-align:center; margin-top:12px; font-size:13px; color:#718096">
      <a href="#" style="color:#4a90e2">Create account</a>
      &nbsp;&#183;&nbsp;
      <a href="#" style="color:#4a90e2">Forgot password?</a>
    </p>
  </div>
  <div id="{eid}-p2" style="display:none; {card}">
    <h2 style="margin:0 0 8px; color:#1a202c; text-align:center">
      Enter your password
    </h2>
    <p style="color:#718096; text-align:center; margin-bottom:24px; font-size:14px">
      Step 2 of 3 — Password verification
    </p>
    <input type="password" placeholder="Password"
      style="width:100%; padding:10px 12px; border:1px solid #cbd5e0;
             border-radius:6px; margin-bottom:16px;
             box-sizing:border-box; font-size:14px;">
    <button onclick="{to_p3}"
      style="width:100%; padding:12px; background:#3182ce; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Sign In
    </button>
  </div>
  <div id="{eid}-p3" style="display:none; {card} text-align:center;">
    <div style="font-size:48px; margin-bottom:16px">&#128075;</div>
    <h2 style="margin:0 0 8px; color:#1a202c">Welcome back!</h2>
    <p style="color:#718096; margin-bottom:28px; font-size:14px">
      Step 3 of 3 — Signed in successfully.
    </p>
    <button class="dismiss-btn" onclick="{dismiss}"
      style="width:100%; padding:12px; background:#38a169; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Continue to Site
    </button>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id}-p3 .dismiss-btn"


@dataclass
class SMSVerificationWall(FrictionElement):
    """Three-phase SMS verification: phone entry → code entry → confirmed.

    Phases:
        1. Phone number entry — click *Send Code*.
        2. 6-digit code entry — click *Verify Code*.
        3. Confirmation screen — click *Continue* (dismisses).

    Step cost = 3.
    """

    step_cost: ClassVar[int] = 3
    element_id: str = "pc-sms-verify"

    def render_html(self) -> str:
        eid = self.element_id
        to_p2 = (
            f"document.getElementById('{eid}-p1').style.display='none';"
            f"document.getElementById('{eid}-p2').style.display='flex'"
        )
        to_p3 = (
            f"document.getElementById('{eid}-p2').style.display='none';"
            f"document.getElementById('{eid}-p3').style.display='flex'"
        )
        dismiss = f"document.getElementById('{eid}').style.display='none'"
        digit_box = (
            '<input type="text" maxlength="1"'
            ' style="width:40px; height:48px; border:2px solid #cbd5e0;'
            " border-radius:6px; text-align:center; font-size:22px;"
            ' font-weight:700; color:#1a202c;">'
        )
        digit_boxes = digit_box * 6
        return f"""
<div id="{eid}" style="
  position:fixed; top:0; left:0; right:0; bottom:0; z-index:9992;
  background:rgba(0,0,0,0.70); display:flex; align-items:center;
  justify-content:center; font-family:sans-serif;">
  <div id="{eid}-p1" style="background:#fff; border-radius:12px; padding:36px;
    max-width:400px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <div style="font-size:40px; margin-bottom:16px">&#128241;</div>
    <h2 style="margin:0 0 8px; color:#1a202c; font-size:20px">Verify Your Phone</h2>
    <p style="color:#718096; margin-bottom:24px; font-size:14px">
      We'll send a verification code to confirm your identity.
    </p>
    <div style="display:flex; gap:8px; margin-bottom:20px;">
      <select style="padding:10px 8px; border:1px solid #cbd5e0; border-radius:6px;
        font-size:14px; background:#f7fafc;">
        <option>+1</option><option>+44</option>
        <option>+86</option><option>+91</option>
      </select>
      <input type="tel" placeholder="(555) 000-0000"
        style="flex:1; padding:10px 12px; border:1px solid #cbd5e0;
               border-radius:6px; font-size:14px;">
    </div>
    <button onclick="{to_p2}"
      style="width:100%; padding:12px; background:#4a90e2; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Send Code
    </button>
  </div>
  <div id="{eid}-p2" style="display:none; background:#fff; border-radius:12px;
    padding:36px; max-width:420px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <div style="font-size:40px; margin-bottom:16px">&#128172;</div>
    <h2 style="margin:0 0 8px; color:#1a202c; font-size:20px">
      Enter Verification Code
    </h2>
    <p style="color:#718096; margin-bottom:24px; font-size:14px">
      Enter the 6-digit code sent to your phone.
    </p>
    <div style="display:flex; gap:8px; justify-content:center; margin-bottom:20px;">
      {digit_boxes}
    </div>
    <button onclick="{to_p3}"
      style="width:100%; padding:12px; background:#4a90e2; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Verify Code
    </button>
    <p style="margin-top:12px; font-size:13px; color:#718096">
      Didn't receive it?
      <a href="#" style="color:#4a90e2">Resend</a>
    </p>
  </div>
  <div id="{eid}-p3" style="display:none; background:#fff; border-radius:12px;
    padding:36px; max-width:380px; width:90%; text-align:center;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);">
    <div style="font-size:48px; margin-bottom:16px">&#9989;</div>
    <h2 style="margin:0 0 12px; color:#1a202c; font-size:20px">Phone Verified!</h2>
    <p style="color:#718096; margin-bottom:28px; font-size:14px">
      Your phone number has been confirmed successfully.
    </p>
    <button class="dismiss-btn" onclick="{dismiss}"
      style="width:100%; padding:12px; background:#38a169; color:#fff; border:none;
             border-radius:6px; cursor:pointer; font-size:15px; font-weight:600;">
      Continue
    </button>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id}-p3 .dismiss-btn"


@dataclass
class MultiStepCookieConsent(FrictionElement):
    """Two-phase GDPR consent: main banner → detailed preferences panel.

    Phases:
        1. Compact cookie banner — click *Manage Preferences*.
        2. Detailed category preferences panel — click *Save & Accept* (dismisses).

    Note: Phase 1 has no "Accept All" shortcut so that both paths require
    exactly 2 clicks, keeping the step cost deterministic.

    Step cost = 2.
    """

    step_cost: ClassVar[int] = 2
    element_id: str = "pc-gdpr-consent"

    def render_html(self) -> str:
        eid = self.element_id
        to_p2 = (
            f"document.getElementById('{eid}-p1').style.display='none';"
            f"document.getElementById('{eid}-p2').style.display='block'"
        )
        dismiss = f"document.getElementById('{eid}').style.display='none'"
        return f"""
<div id="{eid}" style="
  position:fixed; bottom:0; left:0; right:0; z-index:9991;
  font-family:sans-serif;">
  <div id="{eid}-p1" style="background:#1a202c; color:#fff; padding:20px 32px;
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:12px;">
    <div style="flex:1; min-width:200px; font-size:14px; line-height:1.5">
      &#127850; We use cookies to personalise content, analyse traffic, and improve
      your experience.
      <a href="#" style="color:#90cdf4">Learn more</a>
    </div>
    <div style="display:flex; gap:10px; flex-shrink:0;">
      <button onclick="{to_p2}"
        style="padding:8px 20px; background:#4a90e2; color:#fff; border:none;
               border-radius:4px; cursor:pointer; font-size:14px;
               white-space:nowrap; font-weight:600;">
        Manage Preferences
      </button>
    </div>
  </div>
  <div id="{eid}-p2" style="display:none; background:#fff;
    border-top:3px solid #4a90e2; padding:24px 32px;
    box-shadow:0 -4px 20px rgba(0,0,0,0.15);">
    <h3 style="margin:0 0 16px; color:#1a202c; font-size:16px">
      Cookie Preferences
    </h3>
    <div style="display:flex; flex-direction:column; gap:12px; margin-bottom:20px;">
      <label style="display:flex; align-items:center; gap:12px;
        cursor:pointer; font-size:14px; color:#4a5568;">
        <input type="checkbox" checked disabled style="width:16px; height:16px;">
        <span>
          <strong>Strictly Necessary</strong> — Required for the site to function
        </span>
      </label>
      <label style="display:flex; align-items:center; gap:12px;
        cursor:pointer; font-size:14px; color:#4a5568;">
        <input type="checkbox" checked style="width:16px; height:16px;">
        <span>
          <strong>Analytics</strong> — Help us understand how visitors use the site
        </span>
      </label>
      <label style="display:flex; align-items:center; gap:12px;
        cursor:pointer; font-size:14px; color:#4a5568;">
        <input type="checkbox" style="width:16px; height:16px;">
        <span>
          <strong>Marketing</strong> — Used to deliver personalised advertisements
        </span>
      </label>
    </div>
    <div style="display:flex; justify-content:flex-end; gap:10px;">
      <button class="dismiss-btn" onclick="{dismiss}"
        style="padding:10px 24px; background:#4a90e2; color:#fff; border:none;
               border-radius:6px; cursor:pointer; font-size:14px; font-weight:600;">
        Save &amp; Accept
      </button>
    </div>
  </div>
</div>
"""

    def get_dismiss_selector(self) -> str:
        return f"#{self.element_id}-p2 .dismiss-btn"


class FrictionStack:
    """Composable stack of friction elements.

    Elements are injected in order; total step_cost is the sum of all
    constituent elements.

    Args:
        elements: Ordered list of FrictionElement instances to stack.
    """

    def __init__(self, elements: list[FrictionElement]):
        self.elements = elements

    @property
    def total_step_cost(self) -> int:
        """Total number of extra steps required to clear this friction stack."""
        return sum(el.step_cost for el in self.elements)

    def inject(self, html: str) -> str:
        """Inject all friction elements into the HTML page in order.

        Args:
            html: Original page HTML.

        Returns:
            HTML with all friction elements injected.
        """
        for element in self.elements:
            html = element.inject(html)
        return html

    @classmethod
    def from_names(cls, names: list[str]) -> "FrictionStack":
        """Create a FrictionStack from a list of element type names.

        Args:
            names: List of strings like ``["cookie_banner", "captcha", "login_wall"]``.

        Returns:
            FrictionStack with the corresponding elements.
        """
        _registry: dict[str, type[FrictionElement]] = {
            # Single-step (step_cost=1)
            "cookie_banner": CookieBanner,
            "captcha": CAPTCHAWall,
            "login_wall": LoginWall,
            "age_verification": AgeVerification,
            # Multi-step (step_cost=3)
            "multi_captcha": MultiStepCAPTCHAWall,
            "multi_login": MultiStepLoginWall,
            "sms_verification": SMSVerificationWall,
            # Multi-step (step_cost=2)
            "gdpr_consent": MultiStepCookieConsent,
        }
        elements: list[FrictionElement] = []
        for name in names:
            if name not in _registry:
                raise ValueError(
                    f"Unknown friction element '{name}'. "
                    f"Available: {list(_registry.keys())}"
                )
            elements.append(_registry[name]())
        return cls(elements)
