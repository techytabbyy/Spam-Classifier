import java.io.*;
import java.util.*;

// A classification model that predicts a label given some text-based data. It can be used for
// tasks such as classifying spam emails.
public class Classifier {
    private ClassifierNode overallRoot;

    // Behavior:
    //  - Constructs a classifier from a file containing decision nodes and labels.
    //    The file provides the necessary structure to build the classifier. 
    //    Each decision node in the file has a "Feature:" line followed by a "Threshold:" line. 
    //    Label nodes contain only a label string. This format matches exactly what is 
    //    produced by the save() method. The file is read using a pre-order traversal.
    // Parameters:
    //  - input: Scanner connected to the file to be used for building the classifier
    // Exceptions:
    //  - IllegalArgumentException: if the provided input is null.
    public Classifier(Scanner input) {
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        this.overallRoot = createTree(input);
    }

    // Behavior:
    //  - Recursively builds the classification tree from the given Scanner input.
    //    Decision nodes contain a feature and threshold, while label nodes are leaf nodes.
    // Parameters:
    //  - input: Scanner for reading the file content used to construct the tree.
    // Returns:
    //  - ClassifierNode: the root of the decision tree
    private static ClassifierNode createTree(Scanner input) {
        if (input.hasNextLine()) {
            String nextLine = input.nextLine();
            if (nextLine.startsWith("Feature:")) {
                String feature = (nextLine.substring("Feature: ".length()));
                String tLine = input.nextLine();
                double threshold = Double.parseDouble(tLine.substring("Threshold: ".length()));
                ClassifierNode left = createTree(input);
                ClassifierNode right = createTree(input);
                return new ClassifierNode(feature, threshold, left, right);
            } else { 
                String label = nextLine;
                return new ClassifierNode(label);
            }
        }
        return null;
    }

    // Behavior:
    //  - Constructs a Classifier from a list of words and their corresponding probability within
    //    a dataset and the correct labels for that same data that the model will be trained on.
    // Parameters:
    //  - data: list of TextBlocks containing words and their associated probabilities
    //  - labels: list of label strings, such as "spam" or "ham", corresponding to the data
    // Exceptions:
    //  - IllegalArgumentException: if either data or labels are null, data and labels are 
    //    different sizes, or if they are empty
    public Classifier(List<TextBlock> data, List<String> labels) {
        if (data == null || labels == null) {
            throw new IllegalArgumentException("Neither data or labels can be null");
        }
        if (data.size() != labels.size()){
            throw new IllegalArgumentException("Data and label lists must be the same size.");
        }
        if (data.isEmpty() || labels.isEmpty()) {
            throw new IllegalArgumentException("Neither data or labels can be empty");
        }

        for (int i = 0; i < data.size(); i++) {
            overallRoot = createTreeFromTwoLists(data.get(i), labels.get(i), overallRoot);
        }
    }

    // Behavior:
    //  - Builds a classification tree based on data and labels. If a leaf node's label doesn't 
    //  match the expected label, a new decision node is created. The findBiggestDifference 
    //  method is called to determine the feature and calculate the midpoint threshold for new 
    //  nodes. The tree is traversed by comparing the feature's probability to the threshold. 
    //  If the probability is less than the threshold, moves left. If it is greater than or 
    //  equal to the threshold, moves right. 
    // Parameters:
    //  - currData: the current TextBlock that is being checked against the tree
    //  - currLabel: The expected label for the current TextBlock
    //  - root: The current node in the tree that is being processed
    // Returns:
    //  - ClassifierNode: the updated root of the tree
    private static ClassifierNode createTreeFromTwoLists(TextBlock currData, String currLabel,
                                                         ClassifierNode root) {
        if (root == null) {
            root = new ClassifierNode(currLabel, currData);
        }
        if (root.label != null) { 
            if (!root.label.equals(currLabel)) { 
                String mostDifferingFeature = currData.findBiggestDifference(root.associatedData);
                double midpoint = midpoint(currData.get(mostDifferingFeature),
                                                root.associatedData.get(mostDifferingFeature));
                ClassifierNode newNode = new ClassifierNode(mostDifferingFeature,
                                                            midpoint, null, null);
                if (currData.get(mostDifferingFeature) < midpoint) { 
                    newNode.left = new ClassifierNode(currLabel, currData);
                    newNode.right = root;
                } else { 
                    newNode.right = new ClassifierNode(currLabel, currData);
                    newNode.left = root;
                }
                return newNode;
            }
        } else { 
            if (currData.containsFeature(root.feature)) { 
                if (currData.get(root.feature) < root.threshold) { 
                    root.left = createTreeFromTwoLists(currData, currLabel, root.left);
                } else {
                    root.right = createTreeFromTwoLists(currData, currLabel, root.right);
                }
            } else { 
                root.left = createTreeFromTwoLists(currData, currLabel, root.left);
            }
        }
        return root;
    }

    // Behavior:
    //  Saves the current classifier tree to a file using pre-order traversal. Decision nodes 
    //  are printed with the feature and threshold, while label nodes are printed with the 
    //  classification.
    // Parameters:
    //  - output: the PrintStream connected to the file the classifier tree
    // Exceptions:
    //  - IllegalArgumentException: if the output is null
    public void save(PrintStream output) {
        if (output == null) {
            throw new IllegalArgumentException("output cannot be null");
        } 
        save(output, overallRoot);
    }

    // Behavior:
    //  - Recursively traverses the classifier tree and writes its contents to the given output
    //    file by pre-order traversal.
    // Parameters:
    //  - output: the PrintStream connected to the file to save to
    //  - root: The current node being processed
    private void save(PrintStream output, ClassifierNode root) {
        if (root != null) {
            if (root.label != null) {
                output.println(root.label);
            } else {
                output.println("Feature: " + root.feature);
                output.println("Threshold: " + root.threshold);
            }
            save(output, root.left);
            save(output, root.right);    
        }
    }

    // Behavior:
    //  - Determines the classification category for the given text data using the
    //    trained model. Returns a label (such as "spam" or "ham") that best matches
    //    the input based on its features.
    // Parameters:
    //  - input: TextBlock piece of data to classify
    // Returns:
    //  - String: The predicted label
    // Exceptions:
    //  - IllegalArgumentException: If the input is null
    public String classify(TextBlock input) {
        if (input == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        return classify(input, overallRoot);
    }

    // Behavior:
    //  - Recursively traverses the tree and classifies the given data. Decisions are based
    //    on whether the feature probability in the input is below or above the threshold.
    // Parameters:
    //  - input: TextBlock to be classified
    //  - root: The current node in the decision tree being processed
    // Returns:
    //  - String: the classification for the input TextBlock
    private String classify(TextBlock input, ClassifierNode root) {
        if (root.label != null) { 
            return root.label;
        } else if (input.get(root.feature) < root.threshold) {
            return classify(input, root.left); 
        } else {
            return classify(input, root.right);
        }
    }

    // Represents a node in the classification tree, which can either be a decision node or a 
    // label node. Decision nodes contain a feature and a threshold, while label nodes store
    // the classification label.
    private static class ClassifierNode {
        public final String label; 
        public final String feature; 
        public final double threshold; 
        public final TextBlock associatedData; 
        public ClassifierNode left; 
        public ClassifierNode right; 

        // Constructs a label ClassifierNode with a given label.
        // This node represents a leaf in the tree that stores the classification label.
        public ClassifierNode(String label) {
            this(label, null);
        }

        // Constructs a label ClassifierNode (leaf node) with a given label and the data 
        // that is associated with the classification.
        public ClassifierNode(String label, TextBlock associatedData) {
            this.label = label;
            this.associatedData = associatedData;
            this.feature = null;
            this.threshold = 0;
        }
        
        // Constructs a decision ClassifierNode with a feature and a threshold
        // This node represents a branch in the tree where a feature is compared to a threshold.
        public ClassifierNode(String feature, double threshold, ClassifierNode left,
                              ClassifierNode right) {
            this.feature = feature;
            this.threshold = threshold;
            this.label = null;
            this.left = left;
            this.right = right;
            this.associatedData = null;
        }
    }


    ////////////////////////////////////////////////////////////////////
    // PROVIDED METHODS - **DO NOT MODIFY ANYTHING BELOW THIS LINE!** //
    ////////////////////////////////////////////////////////////////////

    // Helper method to calcualte the midpoint of two provided doubles.
    private static double midpoint(double one, double two) {
        return Math.min(one, two) + (Math.abs(one - two) / 2.0);
    }    

    // Behavior: Calculates the accuracy of this model on provided Lists of 
    //           testing 'data' and corresponding 'labels'. The label for a 
    //           datapoint at an index within 'data' should be found at the 
    //           same index within 'labels'.
    // Exceptions: IllegalArgumentException if the number of datapoints doesn't match the number 
    //             of provided labels
    // Returns: a map storing the classification accuracy for each of the encountered labels when
    //          classifying
    // Parameters: data - the list of TextBlock objects to classify. Should be non-null.
    //             labels - the list of expected labels for each TextBlock object. 
    //             Should be non-null.
    public Map<String, Double> calculateAccuracy(List<TextBlock> data, List<String> labels) {
        // Check to make sure the lists have the same size (each datapoint has an expected label)
        if (data.size() != labels.size()) {
            throw new IllegalArgumentException(
                    String.format("Length of provided data [%d] doesn't match provided labels [%d]",
                                  data.size(), labels.size()));
        }
        
        // Create our total and correct maps for average calculation
        Map<String, Integer> labelToTotal = new HashMap<>();
        Map<String, Double> labelToCorrect = new HashMap<>();
        labelToTotal.put("Overall", 0);
        labelToCorrect.put("Overall", 0.0);
        
        for (int i = 0; i < data.size(); i++) {
            String result = classify(data.get(i));
            String label = labels.get(i);

            // Increment totals depending on resultant label
            labelToTotal.put(label, labelToTotal.getOrDefault(label, 0) + 1);
            labelToTotal.put("Overall", labelToTotal.get("Overall") + 1);
            if (result.equals(label)) {
                labelToCorrect.put(result, labelToCorrect.getOrDefault(result, 0.0) + 1);
                labelToCorrect.put("Overall", labelToCorrect.get("Overall") + 1);
            }
        }

        // Turn totals into accuracy percentage
        for (String label : labelToCorrect.keySet()) {
            labelToCorrect.put(label, labelToCorrect.get(label) / labelToTotal.get(label));
        }
        return labelToCorrect;
    }
}


