import { useEffect, useRef } from "react";
import { getAssetPath } from "@/lib/assets";

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
        {/* Aerospace Image 1 - Left */}
        <div className="absolute top-1/4 left-8 md:left-16 lg:left-24 floating-card">
          <img
            src={getAssetPath("lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png")}
            alt="Ethiopian Airlines Fleet"
            className="w-32 md:w-40 lg:w-48 h-auto transform rotate-12 opacity-80"
            style={{ animationDelay: '0s' }}
          />
        </div>

        {/* Aerospace Image 2 - Right */}
        <div className="absolute top-1/3 right-8 md:right-16 lg:right-24 floating-card">
          <img
            src={getAssetPath("lovable-uploads/7e9814d1-b051-4b58-99a9-b57a50fe4738.png")}
            alt="University of Florida Engineering Lab"
            className="w-32 md:w-40 lg:w-48 h-auto transform -rotate-12 opacity-80"
            style={{ animationDelay: '2s' }}
          />
        </div>

        {/* Aerospace Image 3 - Top Left */}
        <div className="absolute top-16 left-1/4 floating-card hidden lg:block">
          <img
            src={getAssetPath("lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png")}
            alt="Jet Engine Turbine"
            className="w-48 xl:w-64 h-auto transform rotate-6 opacity-70"
            style={{ animationDelay: '1s' }}
          />
        </div>

        {/* Aerospace Image 4 - Bottom Right */}
        <div className="absolute bottom-16 right-1/4 floating-card hidden lg:block">
          <img
            src={getAssetPath("lovable-uploads/8cf36141-768e-42d1-9dd6-1da18d8ddee5.png")}
            alt="Jet Engine Testing"
            className="w-48 xl:w-64 h-auto transform -rotate-6 opacity-70"
            style={{ animationDelay: '3s' }}
          />
        </div>
      </div>

      {/* Main Content */}
      <div ref={heroRef} className="relative z-10 text-center section-padding max-w-5xl mx-auto">
        <div className="mb-6 animate-fade-in">
          <p className="text-sm md:text-base text-muted-foreground font-medium mb-2">
            Aerospace Engineer & Manufacturing Innovation Specialist
          </p>
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full animate-glow-pulse" />
          </div>
        </div>

        <h1 className="hero-text text-4xl md:text-6xl lg:text-7xl xl:text-8xl mb-8 animate-fade-in">
          Engineering tomorrow's{" "}
          <span className="text-gradient">aerospace innovations!</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed animate-fade-in">
          Aerospace Engineering student at University of Florida with hands-on experience in manufacturing engineering, 
          CAD design, and advanced simulation. Passionate about integrating cutting-edge manufacturing into aerospace innovation.
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