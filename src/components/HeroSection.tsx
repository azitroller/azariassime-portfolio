import { useEffect, useRef } from "react";
import mockupMobile1 from "@/assets/mockup-mobile-1.png";
import mockupMobile2 from "@/assets/mockup-mobile-2.png";
import mockupDesktop1 from "@/assets/mockup-desktop-1.png";
import mockupDesktop2 from "@/assets/mockup-desktop-2.png";

const HeroSection = () => {
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (heroRef.current) {
        const scrolled = window.pageYOffset;
        const parallax = scrolled * 0.5;
        heroRef.current.style.transform = `translateY(${parallax}px)`;
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToAbout = () => {
    const aboutElement = document.getElementById('about');
    if (aboutElement) {
      aboutElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="hero" className="min-h-screen relative flex items-center justify-center overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-dark opacity-50" />
      
      {/* Floating UI Mockups */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Mobile Mockup 1 - Left */}
        <div className="absolute top-1/4 left-8 md:left-16 lg:left-24 floating-card">
          <img
            src={mockupMobile1}
            alt="Mobile App Design"
            className="w-32 md:w-40 lg:w-48 h-auto transform rotate-12 opacity-80"
            style={{ animationDelay: '0s' }}
          />
        </div>

        {/* Mobile Mockup 2 - Right */}
        <div className="absolute top-1/3 right-8 md:right-16 lg:right-24 floating-card">
          <img
            src={mockupMobile2}
            alt="Mobile App Design"
            className="w-32 md:w-40 lg:w-48 h-auto transform -rotate-12 opacity-80"
            style={{ animationDelay: '2s' }}
          />
        </div>

        {/* Desktop Mockup 1 - Top Left */}
        <div className="absolute top-16 left-1/4 floating-card hidden lg:block">
          <img
            src={mockupDesktop1}
            alt="Desktop App Design"
            className="w-48 xl:w-64 h-auto transform rotate-6 opacity-70"
            style={{ animationDelay: '1s' }}
          />
        </div>

        {/* Desktop Mockup 2 - Bottom Right */}
        <div className="absolute bottom-16 right-1/4 floating-card hidden lg:block">
          <img
            src={mockupDesktop2}
            alt="Desktop App Design"
            className="w-48 xl:w-64 h-auto transform -rotate-6 opacity-70"
            style={{ animationDelay: '3s' }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div ref={heroRef} className="relative z-10 text-center section-padding max-w-5xl mx-auto">
        <div className="mb-6 animate-fade-in">
          <p className="text-sm md:text-base text-muted-foreground font-medium mb-2">
            Experience Designer & Visual Storyteller
          </p>
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full animate-glow-pulse" />
          </div>
        </div>

        <h1 className="hero-text text-4xl md:text-6xl lg:text-7xl xl:text-8xl mb-8 animate-fade-in">
          Crafting people friendly{" "}
          <span className="text-gradient">digital journeys!</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed animate-fade-in">
          A Senior UX & UI Designer based in Kuala Lumpur with over 5 years of experience, 
          crafting user-centric fintech and web experiences. Blending product thinking with visual design.
        </p>

        <div className="animate-fade-in">
          <button
            onClick={scrollToAbout}
            className="btn-primary text-lg"
          >
            Get to know me
          </button>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <button
          onClick={scrollToAbout}
          className="w-8 h-12 border-2 border-primary rounded-full flex items-end justify-center pb-2 hover:border-primary/80 transition-colors duration-300"
        >
          <div className="w-1 h-3 bg-primary rounded-full animate-pulse" />
        </button>
      </div>
    </section>
  );
};

export default HeroSection;