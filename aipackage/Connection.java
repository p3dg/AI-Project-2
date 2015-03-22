package aipackage;

/**
 * @author Peter Galvin
 * 
 */
public class Connection {
    
    public Connection(Node from, Node to, double weight) {
        m_from = from;
        m_to = to;
        m_weight = weight;
        m_deltaw_aggregate = 0;
    }
    
    public Node getFromNode() {
        return m_from;
    }
    
    public Node getToNode() {
        return m_to;
    }
    
    public double getWeight() {
        return m_weight;
    }
    
    public void calcDeltaw(double r) {
    	double outputFromNode = this.getFromNode().getOutput();
    	double outputToNode = this.getToNode().getOutput();
    	double betaToNode = this.getToNode().getBeta();
    	m_deltaw = r * outputFromNode * outputToNode * (1.0-outputToNode) * betaToNode;
    }
    
    public void addDeltawToAggregate() {
    	m_deltaw_aggregate += m_deltaw;
    }
    
    public void applyDeltawAggregate() {
    	m_weight += m_deltaw_aggregate;
    	m_deltaw_aggregate = 0;
    }
    
    private double m_weight;
    private double m_deltaw;
    private double m_deltaw_aggregate;

    private Node m_from;
    private Node m_to;

}
