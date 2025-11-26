// ======================================================================
// DARK MODE TOGGLE (FULL WORKING VERSION WITH LOCALSTORAGE)
// ======================================================================
const darkToggle = document.getElementById("darkToggle");

// Load stored theme
const savedTheme = localStorage.getItem("theme");

if (savedTheme === "light") {
    document.documentElement.classList.remove("dark");
    if (darkToggle) darkToggle.checked = false;
} else {
    document.documentElement.classList.add("dark");
    if (darkToggle) darkToggle.checked = true;
}

if (darkToggle) {
    darkToggle.addEventListener("change", () => {
        if (darkToggle.checked) {
            document.documentElement.classList.add("dark");
            localStorage.setItem("theme", "dark");
        } else {
            document.documentElement.classList.remove("dark");
            localStorage.setItem("theme", "light");
        }
    });
}



// ======================================================================
// DOM ELEMENTS
// ======================================================================
const uploadForm = document.getElementById("uploadForm");
const csvInput = document.getElementById("csvInput");
const uploadBtn = document.getElementById("uploadBtn");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");

const fileNameEl = document.getElementById("fileName");
const statusEl = document.getElementById("status");

const placeholderEl = document.getElementById("placeholder");
const chartImageEl = document.getElementById("chartImage");

const progressWrapper = document.getElementById("progressWrapper");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");

const chartsContainer = document.getElementById("chartsContainer");
const chartsEmptyEl = document.getElementById("chartsEmpty");

const datasetSummary = document.getElementById("datasetSummary");
const sentimentSummary = document.getElementById("sentimentSummary");
const entitySummary = document.getElementById("entitySummary");
const topicCardsContainer = document.getElementById("topicCardsContainer");

const searchBtn = document.getElementById("searchBtn");
const searchInput = document.getElementById("searchInput");
const searchResults = document.getElementById("searchResults");


// ======================================================================
// ENDPOINTS
// ======================================================================
const API_ANALYZE = "/api/analyze/";
const API_REPORT = "/api/report/";
const API_SEARCH = "/api/search/";

let PIPELINE_DATA = null;


// ======================================================================
// FILE NAME DISPLAY
// ======================================================================
csvInput.addEventListener("change", () => {
    if (csvInput.files && csvInput.files[0]) {
        fileNameEl.textContent = `Selected: ${csvInput.files[0].name}`;
    } else {
        fileNameEl.textContent = "";
    }
});


// ======================================================================
// PROGRESS BAR UPDATE
// ======================================================================
function setProgress(percent, text) {
    progressWrapper.classList.remove("hidden");
    progressBar.style.width = percent + "%";
    progressText.textContent = text;
}


// ======================================================================
// CLEAR UI
// ======================================================================
function clearUI() {
    chartImageEl.classList.add("hidden");
    placeholderEl.classList.remove("hidden");

    chartsContainer.innerHTML = "";
    chartsEmptyEl.classList.remove("hidden");

    datasetSummary.innerHTML = "";
    sentimentSummary.innerHTML = "";
    entitySummary.innerHTML = "";
    topicCardsContainer.innerHTML = "";

    setProgress(5, "Starting…");
}


// ======================================================================
// FORM SUBMISSION
// ======================================================================
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!csvInput.files || !csvInput.files[0]) {
        statusEl.textContent = "Please select a CSV or XLSX file.";
        statusEl.className = "text-xs text-rose-400";
        return;
    }

    const formData = new FormData();
    formData.append("file", csvInput.files[0]);

    clearUI();
    setProgress(15, "Uploading file…");

    try {
        const responsePromise = fetch(API_ANALYZE, {
            method: "POST",
            body: formData,
        });

        // FAKE REALTIME PROGRESS
        setTimeout(() => setProgress(35, "Preprocessing text…"), 400);
        setTimeout(() => setProgress(55, "Generating embeddings…"), 1200);
        setTimeout(() => setProgress(75, "Clustering topics…"), 2400);
        setTimeout(() => setProgress(90, "Building visualizations…"), 3200);

        const response = await responsePromise;
        setProgress(100, "Done!");

        const data = await response.json();
        PIPELINE_DATA = data;

        placeholderEl.classList.add("hidden");

        renderResultsFromJson(data);

        statusEl.textContent = "Analysis complete.";
        statusEl.className = "text-xs text-emerald-300";

        downloadPdfBtn.disabled = false;

    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.className = "text-xs text-rose-400";
        downloadPdfBtn.disabled = true;
    }
});



// ======================================================================
// RENDER RESULTS
// ======================================================================
document.querySelectorAll(".summary-toggle").forEach(btn => {
    btn.addEventListener("click", () => {
        const id = btn.getAttribute("data-target");
        document.getElementById(id).classList.toggle("hidden");
    });
});
function renderResultsFromJson(data) {
    chartsContainer.innerHTML = "";
    chartsEmptyEl.classList.add("hidden");

    // Main chart
    if (data.charts.length > 0) {
        const img = data.charts[0].imageBase64;
        if (img && img.length > 50) {
            chartImageEl.src = `data:image/png;base64,${img}`;
            chartImageEl.classList.remove("hidden");
        }
    }

    // Other charts
    data.charts.slice(1).forEach((c) => {
        if (!c.imageBase64 || c.imageBase64.length < 50) return;

        const div = document.createElement("div");
        div.className =
            "rounded-xl border border-slate-800 bg-slate-900/70 p-3 shadow";

        div.innerHTML = `
              <p class="text-xs font-medium text-slate-200">${c.title}</p>
              <img src="data:image/png;base64,${c.imageBase64}" class="mt-2 rounded" />
        `;

        chartsContainer.appendChild(div);
    });

    // Dataset summary
    datasetSummary.innerHTML = `
        <p>Documents: ${data.dataset_insights.num_documents}</p>
        <p>Average Length: ${data.dataset_insights.avg_length}</p>
        <p>Median Length: ${data.dataset_insights.median_length}</p>
    `;

    // Sentiment
    sentimentSummary.innerHTML =
        `<pre>${JSON.stringify(data.summary.sentiment_overall, null, 2)}</pre>`;

    // Entities
    entitySummary.innerHTML =
        `<pre>${JSON.stringify(data.summary.entity_overall, null, 2)}</pre>`;

    // Topic cards
    topicCardsContainer.innerHTML = "";
    for (const [topicName, info] of Object.entries(data.topic_cards)) {
        const card = document.createElement("div");
        card.className =
            "rounded-xl border border-slate-800 bg-slate-900/70 p-4 shadow-xl";

        card.innerHTML = `
            <p class="text-emerald-300 font-semibold">${topicName}</p>
            <p class="mt-2 text-sm">${info.summary}</p>
        `;

        topicCardsContainer.appendChild(card);
    }
}



// ======================================================================
// SEMANTIC SEARCH
// ======================================================================
searchBtn.addEventListener("click", async () => {
    const query = searchInput.value.trim();
    if (!query) return;

    searchResults.innerHTML = "Searching…";

    const response = await fetch(API_SEARCH, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
    });

    const data = await response.json();

    searchResults.innerHTML = data.results
        .map((r) => `<div class="p-2 bg-slate-800/60 rounded">${r.text}</div>`)
        .join("");
});