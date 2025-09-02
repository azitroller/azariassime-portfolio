import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";

const Navigation = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setIsMobileMenuOpen(false);
    }
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-background/90 backdrop-blur-md border-b border-border' 
          : 'bg-transparent'
      }`}
    >
      <div className="container-max section-padding">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <button
              onClick={() => scrollToSection('hero')}
              className="text-2xl font-bold text-foreground hover:text-primary transition-colors duration-300"
            >
              Azarias.
            </button>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => scrollToSection('about')}
              className="text-foreground hover:text-primary transition-colors duration-300"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection('experience')}
              className="text-foreground hover:text-primary transition-colors duration-300"
            >
              Resume
            </button>
            <button
              onClick={() => scrollToSection('work')}
              className="text-foreground hover:text-primary transition-colors duration-300"
            >
              Projects
            </button>
            <button
              onClick={() => scrollToSection('contact')}
              className="btn-primary"
            >
              Contact
            </button>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="text-foreground hover:text-primary transition-colors duration-300"
            >
              {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-full left-0 right-0 bg-background/95 backdrop-blur-md border-b border-border">
            <div className="section-padding py-4 space-y-4">
              <button
                onClick={() => scrollToSection('about')}
                className="block w-full text-left text-foreground hover:text-primary transition-colors duration-300"
              >
                About
              </button>
              <button
                onClick={() => scrollToSection('experience')}
                className="block w-full text-left text-foreground hover:text-primary transition-colors duration-300"
              >
                Resume
              </button>
              <button
                onClick={() => scrollToSection('work')}
                className="block w-full text-left text-foreground hover:text-primary transition-colors duration-300"
              >
                Projects
              </button>
              <button
                onClick={() => scrollToSection('contact')}
                className="btn-primary w-full"
              >
                Contact
              </button>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;