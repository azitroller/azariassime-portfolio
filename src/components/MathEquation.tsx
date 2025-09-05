import { useEffect, useRef } from 'react';

interface MathEquationProps {
  equation: string;
  variables?: { symbol: string; description: string }[];
  centered?: boolean;
}

export const MathEquation = ({ equation, variables, centered = true }: MathEquationProps) => {
  const mathRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (mathRef.current && (window as any).MathJax) {
      (window as any).MathJax.typesetPromise([mathRef.current]).catch((err: any) => {
        console.error('MathJax typeset error:', err);
      });
    }
  }, [equation]);

  return (
    <div className={`equation-block my-8 p-6 bg-background/50 rounded-lg border ${centered ? 'text-center' : ''}`}>
      <div 
        ref={mathRef}
        className="text-lg mb-4"
        style={{ fontFamily: 'Computer Modern, serif' }}
      >
        {equation.startsWith('$$') ? equation : `$$${equation}$$`}
      </div>
      
      {variables && variables.length > 0 && (
        <div className="mt-4 text-left">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Where:</h4>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {variables.map((variable, index) => (
              <li key={index} className="flex">
                <span className="font-mono font-semibold mr-2 min-w-[2rem]">{variable.symbol}</span>
                <span>= {variable.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};