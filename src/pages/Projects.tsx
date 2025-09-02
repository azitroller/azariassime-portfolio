import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Navigation from "@/components/Navigation";
import ContactSection from "@/components/ContactSection";
import Footer from "@/components/Footer";
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from "@/components/ui/carousel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import Autoplay from "embla-carousel-autoplay";

const projects = [
  {
    id: "automated-valve-test",
    name: "Automated Valve Test Platform",
    description: "Developed a Python-driven automation system to test aerospace grade valves under extreme pressure and temperature conditions.",
    tags: ["High-Pressure Testing", "Python", "Automation"]
  },
  {
    id: "rga-sensor-integration", 
    name: "RGA Sensor Integration with Unitree Go2 Robot",
    description: "Designed and simulated a robust mounting system to integrate a Residual Gas Analyzer sensor onto a quadruped robot.",
    tags: ["CAD Design", "Vibration Isolation", "Robotics Integration"]
  },
  {
    id: "uav-propulsion-optimization",
    name: "UAV Propulsion Optimization via High-Fidelity Simulation", 
    description: "Conducted advanced CFD, combustion, and acoustic simulations of UAV propulsion systems for performance optimization.",
    tags: ["ANSYS Fluent", "CFD Modeling", "LMS Virtual.Lab"]
  },
  {
    id: "vibration-fatigue-detection",
    name: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
    description: "Developed real-time anomaly detection algorithms using FFT analysis and machine learning for fatigue risk assessment.",
    tags: ["FFT Analysis", "Machine Learning", "Real-Time Detection"]
  },
  {
    id: "uav-tail-fuselage",
    name: "UAV Tail & Fuselage Variations for Stability Analysis",
    description: "Performed comprehensive stability analysis of various UAV design configurations to optimize flight performance.",
    tags: ["Stability Analysis", "Flight Dynamics", "Design Optimization"]
  }
];

const Projects = () => {
  const navigate = useNavigate();
  const [api, setApi] = useState<any>();
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (!api) {
      return;
    }

    setCurrent(api.selectedScrollSnap());

    api.on("select", () => {
      setCurrent(api.selectedScrollSnap());
    });
  }, [api]);

  const handleProjectClick = (projectId: string) => {
    navigate(`/projects/${projectId}`);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="pt-24 pb-16">
        <div className="container-max section-padding">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-16">
              <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
                Projects
              </h1>
              <p className="text-lg text-muted-foreground">
                Aerospace engineering solutions & innovations
              </p>
            </div>

            <div className="relative">
              <Carousel
                setApi={setApi}
                className="w-full"
                plugins={[
                  Autoplay({
                    delay: 3000,
                    stopOnInteraction: true,
                  }),
                ]}
              >
                <CarouselContent>
                  {projects.map((project) => (
                    <CarouselItem key={project.id}>
                      <Card className="border-border bg-card/50 backdrop-blur-sm">
                        <CardContent className="p-8 md:p-12">
                          <div className="text-center space-y-6">
                            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
                              {project.name}
                            </h2>
                            
                            <p className="text-muted-foreground text-lg max-w-3xl mx-auto leading-relaxed">
                              {project.description}
                            </p>
                            
                            <div className="flex flex-wrap gap-2 justify-center">
                              {project.tags.map((tag) => (
                                <Badge 
                                  key={tag} 
                                  variant="secondary"
                                  className="text-sm px-3 py-1"
                                >
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                            
                            <Button 
                              onClick={() => handleProjectClick(project.id)}
                              className="mt-8"
                              size="lg"
                            >
                              View Details
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </CarouselItem>
                  ))}
                </CarouselContent>
                
                <CarouselPrevious className="left-4" />
                <CarouselNext className="right-4" />
              </Carousel>
              
              {/* Dot indicators */}
              <div className="flex justify-center mt-8 space-x-2">
                {projects.map((_, index) => (
                  <button
                    key={index}
                    className={`w-3 h-3 rounded-full transition-all duration-300 ${
                      index === current
                        ? "bg-primary scale-110"
                        : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
                    }`}
                    onClick={() => api?.scrollTo(index)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      <ContactSection />
      <Footer />
    </div>
  );
};

export default Projects;