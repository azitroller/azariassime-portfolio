import { useEffect, useRef, useState } from "react";
import { ExternalLink, ArrowUpRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

const WorkSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const navigate = useNavigate();

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

  const projects = [
    {
      title: "Automated Valve Test Platform",
      description: "Developed a Python-driven automation system to test aerospace grade valves under extreme pressure and temperature conditions.",
      category: "Aerospace Testing",
      year: "2024",
      technologies: ["High-Pressure", "High-Temperature Testing", "Python"]
    },
    {
      title: "RGA Sensor Integration with Unitree Go2 Robot",
      description: "Designed and simulated a robust mounting system to integrate a Residual Gas Analyzer sensor onto a quadruped robot.",
      category: "Robotics Integration",
      year: "2024",
      technologies: ["CAD Design & Simulation", "Vibration Isolation", "Robotics Integration"]
    },
    {
      title: "UAV Propulsion Optimization via High-Fidelity Simulation",
      description: "Conducted advanced CFD, combustion, and acoustic simulations of UAV propulsion systems.",
      category: "Aerospace Simulation",
      year: "2023",
      technologies: ["ANSYS Fluent", "LMS Virtual.Lab", "CFD & Combustion Modeling"]
    },
    {
      title: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
      description: "Developed an advanced algorithm using FFT and machine learning to detect fatigue risks in real-time for NASA's mass spectrometer systems.",
      category: "NASA Research",
      year: "2023",
      technologies: ["FFT", "Machine Learning", "Real-Time Anomaly Detection"]
    }
  ];

  return (
    <section
      id="work"
      ref={sectionRef}
      className="py-24 md:py-32 section-padding"
    >
      <div className="container-max">
        {/* Section Header */}
        <div className={`text-center mb-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <h2 className="text-sm font-medium text-primary mb-4 tracking-wide uppercase">
            Projects
          </h2>
          <h3 className="text-heading text-3xl md:text-4xl lg:text-5xl font-bold">
            Design highlights
          </h3>
        </div>

        {/* Projects Grid */}
        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          {projects.map((project, index) => (
            <div
              key={project.title}
              className={`group cursor-pointer ${
                isVisible ? 'animate-fade-in' : 'opacity-0'
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="bg-card border border-border rounded-2xl p-8 hover:border-primary/50 transition-all duration-500 hover:shadow-soft hover:scale-105">
                {/* Project Header */}
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <span className="text-sm font-medium text-primary">
                      {project.category}
                    </span>
                    <p className="text-sm text-muted-foreground mt-1">
                      {project.year}
                    </p>
                  </div>
                  <ArrowUpRight 
                    className="w-6 h-6 text-muted-foreground group-hover:text-primary group-hover:scale-110 transition-all duration-300" 
                  />
                </div>

                {/* Project Content */}
                <h4 className="text-xl md:text-2xl font-bold mb-4 group-hover:text-primary transition-colors duration-300">
                  {project.title}
                </h4>
                
                <p className="text-muted-foreground leading-relaxed mb-6">
                  {project.description}
                </p>

                {/* Technologies */}
                <div className="flex flex-wrap gap-2">
                  {project.technologies.map((tech) => (
                    <span
                      key={tech}
                      className="px-3 py-1 bg-muted text-muted-foreground rounded-lg text-sm font-medium"
                    >
                      {tech}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* View All Projects CTA */}
        <div className={`text-center mt-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <p className="text-muted-foreground mb-6">
            Want to see more of my work?
          </p>
          <button 
            onClick={() => navigate('/projects')}
            className="btn-secondary inline-flex items-center gap-2"
          >
            View All Projects
            <ExternalLink className="w-4 h-4" />
          </button>
        </div>
      </div>
    </section>
  );
};

export default WorkSection;