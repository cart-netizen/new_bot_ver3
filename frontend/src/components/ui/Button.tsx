import { cn } from '../../utils/helpers.ts';

interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline';
  className?: string;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  onClick?: () => void;
  children?: React.ReactNode;
}

export function Button({
  className,
  variant = 'default',
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        'px-4 py-2 rounded-lg font-medium transition-colors',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variant === 'default' && 'bg-primary text-white hover:bg-primary/90',
        variant === 'destructive' && 'bg-danger text-white hover:bg-danger/90',
        variant === 'outline' && 'border border-gray-700 text-white hover:bg-gray-800',
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}