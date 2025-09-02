import { useParams, Link } from "react-router-dom";
import Navigation from "@/components/Navigation";
import { ArrowLeft } from "lucide-react";

const ProjectDetail = () => {
  const { id } = useParams();

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="pt-24 pb-16">
        <div className="container-max section-padding">
          <div className="max-w-4xl mx-auto">
            <Link 
              to="/projects"
              className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8"
            >
              <ArrowLeft size={16} />
              Back to Projects
            </Link>

            <div className="text-center py-20">
              <h1 className="text-4xl font-bold text-foreground mb-4">
                Project Details
              </h1>
              <p className="text-muted-foreground mb-8">
                Project ID: {id}
              </p>
              <p className="text-muted-foreground">
                Detailed project information coming soon...
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ProjectDetail;