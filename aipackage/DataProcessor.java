package aipackage;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.Locale;

public class DataProcessor {
    static String[] A1_cand = { "b", "a", "?" };
    static String[] A4_cand = { "u", "y", "l", "t", "?" };
    static String[] A5_cand = { "g", "p", "gg", "?" };
    static String[] A6_cand = { "c", "d", "cc", "i", "j", "k", "m", "r", "q",
            "w", "x", "e", "aa", "ff", "?" };
    static String[] A7_cand = { "v", "h", "bb", "j", "n", "z", "dd", "ff", "o", "?" };
    static String[] A9_cand = { "t", "f", "?" };
    static String[] A10_cand = { "t", "f", "?" };
    static String[] A12_cand = { "t", "f", "?" };
    static String[] A13_cand = { "g", "p", "s", "?" };

    public double[][] m_inputvs;
    public double[][] m_outputvs;

    private List<CreditData> m_datas;

    class CreditData {
        public CreditData(double[] inputs, double[] outputs) {
            m_inputs = inputs;
            m_outputs = outputs;
        }

        public double[] m_inputs;
        public double[] m_outputs;
    }

    double cvtDouble(String [] candidates, String name) {
        for (int i = 0; i < candidates.length; ++i) {
            if (candidates[i].equals(name)) {
                return i;
            }
        }
        return candidates.length;
    }

    public DataProcessor(String aFileName) throws FileNotFoundException {
        m_datas = new ArrayList<CreditData>();
        Scanner s = null;
        
        FileReader f = new FileReader(aFileName);
        
        try {
            s = new Scanner(new BufferedReader(f));
            s.useLocale(Locale.US);

            while (s.hasNextLine()) {
                CreditData data = processLine(s.nextLine());
                m_datas.add(data);
            }
        } finally {
            s.close();
        }

        int i = 0;
        m_inputvs = new double[m_datas.size()][];
        m_outputvs = new double[m_datas.size()][];
        for (CreditData data: m_datas) {
            m_inputvs[i] = data.m_inputs;
            m_outputvs[i] = data.m_outputs;
            ++i;
        }
        
        int numInputs = m_datas.size();
        // columnTotal is a vector of the sums of each "column" of data, that is
        // it sums over a single element at a time over all the input vectors
        double[] columnTotal = new double[m_inputvs[0].length];
        
        for(int j = 0; j < m_inputvs[0].length; j++) {
        	// Compute columnTotal
        	for(int k = 0; k < numInputs; k++) {
                columnTotal[j] += m_inputvs[k][j];
            }
        	
            // Compute average and stddev
            double ave = columnTotal[j]/numInputs;
            // For stddev:
            //  for each element from a column:
            //  subtract mean from element
            //  square result
            //  sum results / numberFeatures
            //  take square root
            double sumSquareDifs = 0;
            for(int k = 0; k < numInputs; k++) {
                sumSquareDifs += Math.pow((m_inputvs[k][j] - ave), 2);
            }
            double stddev = Math.sqrt(sumSquareDifs/numInputs);
            
            for(int k = 0; k < numInputs; k++) {
                m_inputvs[k][j] = (m_inputvs[k][j]-ave)/stddev;
            }
        }
    }

    private double nextDouble(Scanner s) {
        if (s.hasNextDouble()) {
            return s.nextDouble();
        } else {
            s.next();
            return 0.0;
        }
    }

    public CreditData processLine(String line) {
        Scanner scanner = new Scanner(line);
        scanner.useDelimiter(",");

        double[] inputs = new double[15];
        double[] outputs = new double[1];

        inputs[0] = cvtDouble(A1_cand, scanner.next());
        inputs[1] = nextDouble(scanner);
        inputs[2] = nextDouble(scanner);
        inputs[3] = cvtDouble(A4_cand, scanner.next());
        inputs[4] = cvtDouble(A5_cand, scanner.next());
        inputs[5] = cvtDouble(A6_cand, scanner.next());
        inputs[6] = cvtDouble(A7_cand, scanner.next());
        inputs[7] = nextDouble(scanner);
        inputs[8] = cvtDouble(A9_cand, scanner.next());
        inputs[9] = cvtDouble(A10_cand, scanner.next());
        inputs[10] = nextDouble(scanner);
        inputs[11] = cvtDouble(A12_cand, scanner.next());
        inputs[12] = cvtDouble(A13_cand, scanner.next());
        inputs[13] = nextDouble(scanner);
        inputs[14] = nextDouble(scanner);

        String output = scanner.next();
        if (output.equals("+")) {
            outputs[0] = 1.0;
        } else {
            outputs[0] = 0.0;
        }
        return new CreditData(inputs, outputs);
    }
    
}



