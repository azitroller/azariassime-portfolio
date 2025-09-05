import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Copy, Expand } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface CodePreviewProps {
  title: string;
  preview: string;
  fullCode: string;
  language?: string;
}

export const CodePreview = ({ title, preview, fullCode, language = 'python' }: CodePreviewProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      toast({
        title: "Code copied to clipboard",
        duration: 2000,
      });
    } catch (err) {
      toast({
        title: "Failed to copy code",
        variant: "destructive",
        duration: 2000,
      });
    }
  };

  return (
    <div className="bg-[#0D1117] rounded-lg border border-gray-800 overflow-hidden shadow-lg">
      <div className="flex items-center justify-between px-4 py-2 bg-[#21262d] border-b border-gray-800">
        <span className="text-sm font-medium text-gray-300">{title}</span>
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => copyToClipboard(preview)}
            className="text-gray-400 hover:text-white h-8 w-8 p-0"
          >
            <Copy className="h-4 w-4" />
          </Button>
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-400 hover:text-white h-8 w-8 p-0"
              >
                <Expand className="h-4 w-4" />
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-5xl max-h-[80vh] bg-[#0D1117] border-gray-800">
              <DialogHeader className="border-b border-gray-800 pb-4">
                <DialogTitle className="text-white flex items-center justify-between">
                  {title}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(fullCode)}
                    className="text-gray-400 hover:text-white"
                  >
                    <Copy className="h-4 w-4 mr-2" />
                    Copy All
                  </Button>
                </DialogTitle>
              </DialogHeader>
              <div className="overflow-auto max-h-[60vh]">
                <pre className="text-sm text-gray-100 font-mono p-4 whitespace-pre-wrap">
                  <code className={`language-${language}`}>{fullCode}</code>
                </pre>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>
      <div className="p-4">
        <pre className="text-sm text-gray-100 font-mono whitespace-pre-wrap overflow-x-auto">
          <code className={`language-${language}`}>{preview}</code>
        </pre>
        <div className="mt-3 flex justify-end">
          <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="text-gray-300 border-gray-600 hover:bg-gray-800">
                Expand Full Code
              </Button>
            </DialogTrigger>
          </Dialog>
        </div>
      </div>
    </div>
  );
};