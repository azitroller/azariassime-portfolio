import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";

const projects = [
  {
    id: "automated-valve-test",
    name: "Automated Valve Test Platform",
    description: "Python-driven automation system for aerospace grade valves"
  },
  {
    id: "rga-sensor-integration", 
    name: "RGA Sensor Integration with Unitree Go2 Robot",
    description: "Robust mounting system for Residual Gas Analyzer sensor"
  },
  {
    id: "uav-propulsion-optimization",
    name: "UAV Propulsion Optimization via High-Fidelity Simulation", 
    description: "Advanced CFD, combustion, and acoustic simulations"
  },
  {
    id: "vibration-fatigue-detection",
    name: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
    description: "Real-time anomaly detection using FFT and machine learning"
  },
  {
    id: "uav-tail-fuselage",
    name: "UAV Tail & Fuselage Variations for Stability Analysis",
    description: "Comprehensive stability analysis of UAV design variations"
  }
];

const Projects = () => {
  const [visibleProjects, setVisibleProjects] = useState<number[]>([]);
  const [currentIndex, setCurentIndex] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setInterval(() => {
      if (currentIndex < projects.length) {
        setVisibleProjects(prev => [...prev, currentIndex]);
        setCurentIndex(prev => prev + 1);
      }
    }, 300);

    return () => clearInterval(timer);
  }, [currentIndex]);

  const handleProjectClick = (projectId: string) => {
    navigate(`/projects/${projectId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="pt-24 pb-16">
        <div className="container-max section-padding">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl font-bold text-foreground mb-2">
              Projects
            </h1>
            <p className="text-muted-foreground mb-12">
              Aerospace engineering solutions & innovations
            </p>

            <div className="bg-card border border-border rounded-lg p-8 font-mono">
              <div className="flex items-center gap-2 mb-6 text-sm text-muted-foreground">
                <div className="w-3 h-3 rounded-full bg-destructive"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="ml-4">azarias@aerospace:~/projects</span>
              </div>

              <div className="space-y-3">
                <div className="text-primary text-sm">
                  $ ls -la projects/
                </div>
                
                {projects.map((project, index) => (
                  <div
                    key={project.id}
                    className={`transition-all duration-500 ${
                      visibleProjects.includes(index)
                        ? "opacity-100 translate-y-0"
                        : "opacity-0 translate-y-4"
                    }`}
                  >
                    <button
                      onClick={() => handleProjectClick(project.id)}
                      className="group w-full text-left p-3 rounded border border-transparent hover:border-primary hover:bg-accent/50 transition-all duration-300"
                    >
                      <div className="flex items-start gap-3">
                        <span className="text-primary text-sm mt-1">{'>'}</span>
                        <div className="flex-1">
                          <div className="text-foreground font-medium group-hover:text-primary transition-colors">
                            {project.name}
                          </div>
                          <div className="text-muted-foreground text-sm mt-1">
                            {project.description}
                          </div>
                        </div>
                        <span className="text-muted-foreground text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                          [ENTER]
                        </span>
                      </div>
                    </button>
                  </div>
                ))}

                {visibleProjects.length === projects.length && (
                  <div className="text-primary text-sm animate-fade-in mt-6">
                    $ █
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Projects;