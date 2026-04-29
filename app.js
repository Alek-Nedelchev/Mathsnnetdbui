const SUPABASE_URL = 'https://jeonqxrhsgptutufrwpo.supabase.co';
const SUPABASE_PUBLISHABLE_KEY = 'sb_publishable_m6n5JpTu5HD_2hdJR3CWAQ_0Pv5gLbx';
const EDGE_FUNCTION_URL = `${SUPABASE_URL}/functions/v1/search`;

const header = document.getElementById('header');

let currentLightboxImage = null;
const searchWrapper = document.getElementById('searchWrapper');
const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const resultsEl = document.getElementById('results');
const statusEl = document.getElementById('status');

let hasSearched = false;

searchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (!query) return;

  if (!hasSearched) {
    hasSearched = true;
    header.classList.add('top');
  }

  resultsEl.classList.add('hidden');
  resultsEl.innerHTML = '';
  showStatus('Searching...');

  try {
    const res = await fetch(EDGE_FUNCTION_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_PUBLISHABLE_KEY}`,
        'apikey': SUPABASE_PUBLISHABLE_KEY,
      },
      body: JSON.stringify({ query, count: 10 }),
    });

    if (res.status === 429) {
      showStatus('Rate limit exceeded. Please wait a minute and try again.', true);
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showStatus(err.error || `Error ${res.status}`, true);
      return;
    }

    const data = await res.json();
    hideStatus();

    if (!data.results || data.results.length === 0) {
      showStatus('No results found. Try a different query.');
      return;
    }

    renderResults(data.results);
  } catch (err) {
    showStatus('Network error. Please try again.', true);
  }
});

function parseImages(imagesData) {
  if (!imagesData) return [];
  if (typeof imagesData === 'string') {
    try {
      return JSON.parse(imagesData);
    } catch {
      return [];
    }
  }
  if (Array.isArray(imagesData)) return imagesData;
  return [];
}

function renderResults(results) {
  resultsEl.innerHTML = results.map((r, i) => {
    const similarity = (r.similarity * 100).toFixed(1);
    const topics = Array.isArray(r.topics_flat) ? r.topics_flat : [];
    const problemPreview = escapeHtml(r.problem_markdown || '');
    const needsExpand = problemPreview.length > 500;
    const images = parseImages(r.images_data);
    const hasImages = images.length > 0 || r.has_images;
    const imageCount = images.length || r.num_images || 0;

    return `
      <div class="result-item" style="animation-delay: ${i * 0.05}s">
        <div class="result-meta">
          ${r.country ? `<span class="badge badge-country">${escapeHtml(r.country)}</span>` : ''}
          ${r.competition ? `<span class="badge badge-competition">${escapeHtml(r.competition)}</span>` : ''}
          ${r.language ? `<span class="badge badge-language">${escapeHtml(r.language)}</span>` : ''}
          ${hasImages ? `<span class="badge badge-images"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>${imageCount}</span>` : ''}
          <span class="badge badge-similarity">${similarity}% match</span>
        </div>
        <div class="result-problem${needsExpand ? '' : ' expanded'}" data-id="${escapeHtml(r.id)}">
          ${problemPreview}
        </div>
        ${needsExpand ? `<button class="result-expand" onclick="toggleExpand(this)">Show more</button>` : ''}
        ${images.length > 0 ? `
          <div class="result-images">
            ${images.map((img, idx) => `
              <div class="result-image-thumb" onclick="openLightbox(${i}, ${idx})">
                <img src="data:image/${img.format || 'png'};base64,${img.data}" alt="Problem figure ${idx + 1}" loading="lazy">
                <div class="result-image-overlay">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
                </div>
              </div>
            `).join('')}
          </div>
        ` : ''}
        ${topics.length > 0 ? `
          <div class="result-topics">
            ${topics.map(t => `<span class="topic-tag">${escapeHtml(t)}</span>`).join('')}
          </div>
        ` : ''}
      </div>
    `;
  }).join('');
  resultsEl.classList.remove('hidden');
}

window.openLightbox = function(resultIndex, imageIndex) {
  const result = resultsEl.children[resultIndex];
  if (!result) return;
  const thumbs = result.querySelectorAll('.result-image-thumb img');
  const img = thumbs[imageIndex];
  if (!img) return;
  
  currentLightboxImage = { resultIndex, imageIndex, total: thumbs.length };
  
  const lightbox = document.createElement('div');
  lightbox.className = 'lightbox';
  lightbox.id = 'lightbox';
  lightbox.innerHTML = `
    <div class="lightbox-backdrop" onclick="closeLightbox()"></div>
    <button class="lightbox-close" onclick="closeLightbox()" aria-label="Close">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
    ${thumbs.length > 1 ? `
      <button class="lightbox-nav lightbox-prev" onclick="navigateLightbox(-1)" aria-label="Previous">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
      </button>
      <button class="lightbox-nav lightbox-next" onclick="navigateLightbox(1)" aria-label="Next">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
      </button>
      <div class="lightbox-counter">${imageIndex + 1} / ${thumbs.length}</div>
    ` : ''}
    <div class="lightbox-content">
      <img src="${img.src}" alt="Problem figure ${imageIndex + 1}" class="lightbox-img">
    </div>
  `;
  document.body.appendChild(lightbox);
  document.body.style.overflow = 'hidden';
  
  lightbox.addEventListener('keydown', handleLightboxKey);
  setTimeout(() => lightbox.classList.add('active'), 10);
};

window.closeLightbox = function() {
  const lightbox = document.getElementById('lightbox');
  if (lightbox) {
    lightbox.classList.remove('active');
    setTimeout(() => {
      lightbox.remove();
      document.body.style.overflow = '';
    }, 200);
  }
  currentLightboxImage = null;
};

window.navigateLightbox = function(direction) {
  if (!currentLightboxImage) return;
  const { resultIndex, imageIndex, total } = currentLightboxImage;
  const newIndex = (imageIndex + direction + total) % total;
  closeLightbox();
  openLightbox(resultIndex, newIndex);
};

function handleLightboxKey(e) {
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navigateLightbox(-1);
  if (e.key === 'ArrowRight') navigateLightbox(1);
}

function toggleExpand(btn) {
  let problemEl = btn.previousElementSibling;
  while (problemEl && !problemEl.classList.contains('result-problem')) {
    problemEl = problemEl.previousElementSibling;
  }
  if (!problemEl) return;
  const isExpanded = problemEl.classList.toggle('expanded');
  btn.textContent = isExpanded ? 'Show less' : 'Show more';
}

function showStatus(msg, isError = false) {
  statusEl.innerHTML = isError
    ? msg
    : `<span class="spinner"></span>${msg}`;
  statusEl.classList.toggle('error', isError);
  statusEl.classList.remove('hidden');
}

function hideStatus() {
  statusEl.classList.add('hidden');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// Theme toggle
const themeToggle = document.getElementById('themeToggle');
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
});
