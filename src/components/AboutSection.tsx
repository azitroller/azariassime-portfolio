import { useEffect, useRef, useState } from "react";

const AboutSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const expertise = [
    {
      title: "Advanced Manufacturing",
      description: "Designing precision fixtures, automated testing systems, and innovative manufacturing solutions using industry-leading CAD tools and simulation software.",
      skills: ["Siemens NX", "SolidWorks", "CATIA", "Fusion 360 CAM", "CNC Machining", "Additive Manufacturing"]
    },
    {
      title: "Aerospace Engineering",
      description: "Developing propulsion systems, analyzing aerodynamics, and optimizing designs for supersonic applications through advanced computational fluid dynamics.",
      skills: ["ANSYS Fluent", "COMSOL", "Compressible Flow", "Propulsion Systems", "CFD Analysis", "Thermal Validation"]
    }
  ];

  return (
    <section
      id="about"
      ref={sectionRef}
      className="py-24 md:py-32 section-padding bg-secondary/20"
    >
      <div className="container-max">
        {/* Section Title */}
        <div className={`text-center mb-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <h2 className="text-sm font-medium text-primary mb-4 tracking-wide uppercase">
            What I do
          </h2>
          <h3 className="text-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
            Areas of expertise
          </h3>
        </div>

        {/* Expertise Grid */}
        <div className="grid md:grid-cols-2 gap-12 lg:gap-16">
          {expertise.map((area, index) => (
            <div
              key={area.title}
              className={`${
                isVisible ? 'animate-fade-in' : 'opacity-0'
              }`}
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              <h4 className="text-2xl md:text-3xl font-bold mb-6 text-foreground">
                {area.title}
              </h4>
              
              <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
                {area.description}
              </p>

              <div className="flex flex-wrap gap-3">
                {area.skills.map((skill) => (
                  <span
                    key={skill}
                    className="px-4 py-2 bg-muted text-muted-foreground rounded-full text-sm font-medium hover:bg-primary hover:text-primary-foreground transition-all duration-300 cursor-default"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className={`text-center mt-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <div className="max-w-3xl mx-auto">
            <h3 className="text-2xl md:text-3xl font-bold mb-6">
              Ready to create something amazing together?
            </h3>
            <p className="text-lg text-muted-foreground mb-8">
              I'm passionate about solving complex problems through thoughtful design 
              and creating digital experiences that truly make a difference.
            </p>
            <button
              onClick={() => document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-primary text-lg"
            >
              Let's collaborate
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;