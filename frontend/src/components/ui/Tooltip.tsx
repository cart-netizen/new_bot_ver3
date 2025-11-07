// frontend/src/components/ui/Tooltip.tsx

import { useState } from 'react';
import { HelpCircle } from 'lucide-react';

interface TooltipProps {
  content: string;
  className?: string;
}

export function Tooltip({ content, className = '' }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block">
      <HelpCircle
        className={`h-4 w-4 text-gray-400 hover:text-gray-300 cursor-help ${className}`}
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      />
      {isVisible && (
        <div className="absolute z-50 w-72 p-3 text-sm text-white bg-gray-900 border border-gray-700 rounded-lg shadow-xl left-1/2 -translate-x-1/2 bottom-full mb-2">
          <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-8 border-r-8 border-t-8 border-l-transparent border-r-transparent border-t-gray-700" />
          <p className="leading-relaxed whitespace-pre-line">{content}</p>
        </div>
      )}
    </div>
  );
}
