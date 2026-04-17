import React, { useMemo, useState } from 'react';
import { absoluteDownloadUrl, processVideo } from './services/api';

const DOMAIN_OPTIONS = [
  { value: 'general', label: '일반(Base 모델)' },
  { value: 'social_news', label: '뉴스 / 시사' },
  { value: 'ent', label: '예능 / 인터뷰' },
  { value: 'vacation', label: '여행 / 브이로그' },
  { value: 'politics', label: '정치' },
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [domain, setDomain] = useState('general');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [segments, setSegments] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');

  const filteredSegments = useMemo(() => {
    if (!searchTerm.trim()) return segments;
    const keyword = searchTerm.toLowerCase();
    return segments.filter((seg) => {
      const text = String(seg.corrected || seg.text || '').toLowerCase();
      return text.includes(keyword);
    });
  }, [segments, searchTerm]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setError('');
    setResult(null);
    setSegments([]);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl('');
    }
  };

  const handleProcess = async () => {
    if (!selectedFile) {
      setError('먼저 영상 파일을 선택해 주세요.');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const response = await processVideo(selectedFile, domain === 'general' ? '' : domain);
      const pipelineResult = response?.pipeline_result || null;
      const rawSegments = pipelineResult?.transcription?.segments || [];
      const mappedSegments = rawSegments.map((seg) => ({
        ...seg,
        corrected: seg.text,
      }));

      setResult(pipelineResult);
      setSegments(mappedSegments);
    } catch (err) {
      setError(err.message || '영상 처리 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSegmentChange = (id, value) => {
    setSegments((prev) =>
      prev.map((seg) =>
        seg.id === id ? { ...seg, corrected: value } : seg
      )
    );
  };

  const transcriptionInfo = result?.transcription;
  const subtitleDownloadUrl = absoluteDownloadUrl(result?.downloads?.subtitle_download_url);
  const videoDownloadUrl = absoluteDownloadUrl(result?.downloads?.video_download_url);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Auto Subtitle Service</h1>
          <p>백엔드 /upload/process API 연동 프론트엔드</p>
        </div>
      </header>

      <main className="layout">
        <section className="panel left-panel">
          <h2>1. 영상 업로드</h2>

          <label className="file-box">
            <input type="file" accept="video/*" onChange={handleFileChange} />
            <span>{selectedFile ? selectedFile.name : '영상 파일을 선택하세요'}</span>
          </label>

          <div className="field-group">
            <label htmlFor="domain">2. 도메인 선택</label>
            <select
              id="domain"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            >
              {DOMAIN_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <button className="primary-button" onClick={handleProcess} disabled={isProcessing || !selectedFile}>
            {isProcessing ? '처리 중...' : '자막 생성 시작'}
          </button>

          {error ? <div className="error-box">{error}</div> : null}

          {previewUrl ? (
            <div className="preview-box">
              <h3>영상 미리보기</h3>
              <video src={previewUrl} controls className="video-player" />
            </div>
          ) : null}
        </section>

        <section className="panel center-panel">
          <h2>처리 결과</h2>

          {!result ? (
            <div className="empty-state">
              아직 처리 결과가 없습니다. 파일을 선택하고 자막 생성을 시작하세요.
            </div>
          ) : (
            <>
              <div className="info-grid">
                <div className="info-card">
                  <span>요청 도메인</span>
                  <strong>{transcriptionInfo?.requested_domain || 'general'}</strong>
                </div>
                <div className="info-card">
                  <span>적용 도메인</span>
                  <strong>{transcriptionInfo?.applied_domain || '-'}</strong>
                </div>
                <div className="info-card">
                  <span>사용 어댑터</span>
                  <strong>{transcriptionInfo?.used_adapter || 'base'}</strong>
                </div>
                <div className="info-card">
                  <span>Fallback</span>
                  <strong>{String(Boolean(transcriptionInfo?.fallback_used))}</strong>
                </div>
                <div className="info-card">
                  <span>세그먼트 수</span>
                  <strong>{transcriptionInfo?.segment_count ?? 0}</strong>
                </div>
                <div className="info-card">
                  <span>모델</span>
                  <strong>{transcriptionInfo?.model_name || '-'}</strong>
                </div>
              </div>

              <div className="download-row">
                <a className="secondary-button" href={subtitleDownloadUrl} target="_blank" rel="noreferrer">
                  SRT 다운로드
                </a>
                <a className="secondary-button" href={videoDownloadUrl} target="_blank" rel="noreferrer">
                  자막 영상 다운로드
                </a>
              </div>

              <div className="text-box">
                <h3>전사 전체 텍스트</h3>
                <p>{transcriptionInfo?.full_text || ''}</p>
              </div>
            </>
          )}
        </section>

        <section className="panel right-panel">
          <div className="editor-header">
            <h2>세그먼트 편집</h2>
            <input
              type="text"
              placeholder="자막 검색"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="segment-list">
            {filteredSegments.length === 0 ? (
              <div className="empty-state small">
                편집할 세그먼트가 없습니다.
              </div>
            ) : (
              filteredSegments.map((seg) => (
                <div className="segment-card" key={seg.id}>
                  <div className="segment-time">
                    <span>{Number(seg.start).toFixed(2)}s</span>
                    <span>{Number(seg.end).toFixed(2)}s</span>
                  </div>
                  <div className="segment-original">원문: {seg.text}</div>
                  <textarea
                    value={seg.corrected}
                    onChange={(e) => handleSegmentChange(seg.id, e.target.value)}
                    rows={3}
                  />
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
