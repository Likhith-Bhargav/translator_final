const form = document.getElementById("translateForm");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const monoLink = document.getElementById("monoLink");
const dualLink = document.getElementById("dualLink");
const errorBox = document.getElementById("errorBox");

// Use Dataiku's backend API endpoint
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  loading.classList.remove("hidden");
  result.classList.add("hidden");
  errorBox.classList.add("hidden");

  const fileInput = document.getElementById("file");
  const lang_in = document.getElementById("lang_in").value;
  const lang_out = document.getElementById("lang_out").value;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("lang_in", lang_in);
  formData.append("lang_out", lang_out);

  try {
    const response = await fetch(getWebAppBackendUrl("/translate"), {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    loading.classList.add("hidden");

    if (response.ok) {
      result.classList.remove("hidden");
      monoLink.href = getWebAppBackendUrl("/download/" + data.mono);
      dualLink.href = getWebAppBackendUrl("/download/" + data.dual);
    } else {
      errorBox.textContent = data.error || "Translation failed.";
      errorBox.classList.remove("hidden");
    }
  } catch (err) {
    loading.classList.add("hidden");
    errorBox.textContent = "Backend connection error: " + err.message;
    errorBox.classList.remove("hidden");
  }
});
