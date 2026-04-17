const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

async function parseJsonSafe(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function processVideo(file, domain) {
  if (!file) {
    throw new Error('업로드할 영상 파일을 선택해 주세요.');
  }

  const formData = new FormData();
  formData.append('file', file);
  if (domain) {
    formData.append('domain', domain);
  }

  const response = await fetch(`${API_BASE}/upload/process`, {
    method: 'POST',
    body: formData,
  });

  const data = await parseJsonSafe(response);

  if (!response.ok) {
    const detail = data?.detail || data?.message || '영상 처리 중 오류가 발생했습니다.';
    throw new Error(detail);
  }

  return data;
}

export function absoluteDownloadUrl(url) {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  return `${API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
}
