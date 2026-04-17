import React from 'react';

function Status({ progress }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 bg-white rounded-3xl shadow-xl border border-gray-100 max-w-md w-full">
      <div className="relative w-32 h-32 mb-8">
        <svg className="w-full h-full" viewBox="0 0 100 100">
          <circle className="text-gray-100 stroke-current" strokeWidth="10" cx="50" cy="50" r="40" fill="transparent" />
          <circle className="text-blue-600 stroke-current transition-all duration-500" strokeWidth="10" strokeLinecap="round" cx="50" cy="50" r="40" fill="transparent" strokeDasharray="251.2" strokeDashoffset={251.2 - (251.2 * progress) / 100} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-2xl font-black text-blue-600">{progress}%</div>
      </div>
      <h3 className="text-xl font-bold text-gray-800">AI 분석 중...</h3>
      <p className="text-gray-400 text-sm mt-2 text-center">문맥에 최적화된 자막을 생성하고 있습니다.</p>
    </div>
  );
}
export default Status;