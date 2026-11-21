// Small progressive enhancements. The console works fully without this file.
(function () {
  "use strict";

  var form = document.getElementById("ticket-form");
  var body = document.getElementById("body");
  var button = document.getElementById("analyze");

  if (form && button) {
    // Loading state while the server runs the pipeline.
    form.addEventListener("submit", function () {
      button.classList.add("loading");
      button.setAttribute("aria-busy", "true");
    });
    // Cmd/Ctrl + Enter submits from either field.
    form.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        if (form.requestSubmit) form.requestSubmit(); else form.submit();
      }
    });
  }

  // A sample prefilled the form: put the cursor in the body so the presenter can submit at once.
  if (body && body.value && !document.querySelector(".has-result")) {
    body.focus();
    body.setSelectionRange(body.value.length, body.value.length);
  }

  // Copy the plain draft text (citation markers included).
  document.querySelectorAll("button.copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var src = document.querySelector(btn.getAttribute("data-copy"));
      if (!src) return;
      var done = function () {
        btn.textContent = "Copied";
        btn.classList.add("copied");
        setTimeout(function () { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1500);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(src.value).then(done, function () {});
      } else {
        src.select();
        try { document.execCommand("copy"); done(); } catch (err) { /* ignore */ }
      }
    });
  });

  // Citation click: open and flash the evidence card.
  document.querySelectorAll("a.cite").forEach(function (a) {
    a.addEventListener("click", function () {
      var card = document.querySelector(a.getAttribute("href"));
      if (!card) return;
      var d = card.querySelector("details");
      if (d) d.open = true;
      card.classList.remove("flash");
      void card.offsetWidth;
      card.classList.add("flash");
    });
  });
})();
