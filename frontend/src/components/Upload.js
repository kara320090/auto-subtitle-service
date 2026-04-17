import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Languages, MessageSquare, Plus } from 'lucide-react';

function Upload({ onUploadSuccess, isDark }) {
  const [style, setStyle] = useState('formal');
  const onDrop = useCallback((files) => { if (files[0]) onUploadSuccess(URL.createObjectURL(files[0]), style); }, [onUploadSuccess, style]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'video/*': [] } });

  const cardStyle = isDark 
    ? "bg-brand-card/40 border-gray-800/20" 
    : "bg-white border-gray-200 shadow-xl";

  return (
    <div className={`w-full max-w-5xl flex flex-col gap-10 backdrop-blur-2xl p-12 rounded-[40px] border transition-all duration-300 relative overflow-hidden z-10 ${cardStyle}`}>
      {isDark && <div className="absolute -top-10 -right-10 w-32 h-32 bg-brand-purple/15 rounded-full blur-3xl"></div>}
      
      <div className="text-center space-y-3 relative">
        <h2 className={`text-4xl font-black tracking-tight leading-tight ${isDark ? 'text-white' : 'text-gray-900'}`}>새 프로젝트 생성</h2>
        <p className={`${isDark ? 'text-gray-400' : 'text-gray-500'} max-w-lg mx-auto text-base`}>영상 스타일을 선택하고 파일을 업로드하여 AI 자막을 시작하세요.</p>
      </div>

      <div className="grid grid-cols-2 gap-8 relative">
        <button onClick={() => setStyle('formal')} className={`p-8 rounded-2xl border-2 transition-all flex flex-col items-center gap-5 ${style === 'formal' ? 'border-brand-purple bg-brand-purple/5 text-brand-purple' : isDark ? 'border-gray-800 bg-brand-card text-gray-500' : 'border-gray-100 bg-gray-50 text-gray-400'}`}>
          <Languages size={48} />
          <div className="text-center">
            <div className={`font-bold text-xl ${style === 'formal' ? 'text-brand-purple' : isDark ? 'text-white' : 'text-gray-800'}`}>문어체 (Formal)</div>
            <div className="text-xs mt-1.5 opacity-70">뉴스, 교육, 발표용 최적화</div>
          </div>
        </button>

        <button onClick={() => setStyle('casual')} className={`p-8 rounded-2xl border-2 transition-all flex flex-col items-center gap-5 ${style === 'casual' ? 'border-brand-purple bg-brand-purple/5 text-brand-purple' : isDark ? 'border-gray-800 bg-brand-card text-gray-500' : 'border-gray-100 bg-gray-50 text-gray-400'}`}>
          <MessageSquare size={48} />
          <div className="text-center">
            <div className={`font-bold text-xl ${style === 'casual' ? 'text-brand-purple' : isDark ? 'text-white' : 'text-gray-800'}`}>구어체 (Casual)</div>
            <div className="text-xs mt-1.5 opacity-70">브이로그, 예능, 소셜 영상</div>
          </div>
        </button>
      </div>

      <div {...getRootProps()} className={`relative border-4 border-dashed rounded-[32px] p-24 text-center transition-all cursor-pointer ${isDragActive ? 'border-brand-purple-light bg-brand-purple/10' : isDark ? 'border-gray-800 bg-brand-card/50 hover:border-brand-purple/50' : 'border-gray-200 bg-gray-50 hover:border-brand-purple/30'}`}>
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-5">
          <div className="w-20 h-20 bg-brand-purple text-white rounded-2xl flex items-center justify-center shadow-xl shadow-brand-purple/20">
            <Plus size={40} />
          </div>
          <div>
            <p className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-800'}`}>영상을 여기에 드래그하거나 클릭</p>
            <p className="text-sm text-gray-400 mt-2 uppercase tracking-widest font-bold">MP4, MOV, WEBM (Max 1GB)</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Upload;