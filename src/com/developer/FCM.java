package com.developer;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class FCM {
    public ArrayList<ArrayList<Float>> data;
    public ArrayList<ArrayList<Float>> clusterCenters;
    private float u[][];
    private float u_pre[][];
    private int clusterCount;
    private int iteration;
    private int dimension;
    private int fuzziness;
    private double epsilon;
    public double finalError;

    public FCM(){
        data = new ArrayList<ArrayList<Float>>();
        clusterCenters = new ArrayList<ArrayList<Float>>();
        fuzziness = 2;
        epsilon = 0.01;
    }

    public void run(int clusterNumber, int iter, ArrayList<ArrayList<Float>> data){
        this.clusterCount = clusterNumber;
        this.iteration = iter;
        this.data = data;

        // Algoritma FCM
        // 1 Inisialisasi derajat keanggotaan
        assignInitialMembership();

        for (int i = 0; i < iteration; i++) {
            // 2 Hitung pusat cluster
            calculateClusterCenters();

            // 3 Update derajat keanggotaan
            updateMembershipValues();

            // 4 Cek konvergensi
            finalError = checkConvergence();
            if(finalError <= epsilon)
                break;
        }
    }

    /**
     * in this function we generate random data with specific option
     * @param numberOfData
     * @param dimension
     * @param minRange
     * @param maxRange
     */
    public void createRandomData(int numberOfData, int dimension, int minRange, int maxRange, int clusterCount){
        this.dimension = dimension;
        ArrayList<ArrayList<Integer>> centroids = new ArrayList<ArrayList<Integer>>();
        centroids.add(new ArrayList<Integer>());
        int[] numberOfDataInEachArea = new int[clusterCount];
        int range = maxRange - minRange + 1;
        int step = range / (clusterCount + 1);
        for (int i = 1; i <= clusterCount; i++) {
            centroids.get(0).add(minRange + i * step);
        }

        for (int i = 0; i < dimension - 1; i++) {
            centroids.add((ArrayList<Integer>) centroids.get(0).clone());
        }
        double variance = (centroids.get(0).get(1) - centroids.get(0).get(0))/ 2.5;
        for (int i = 0; i < dimension; i++) {
            Collections.shuffle(centroids.get(i));
        }
        Random r = new Random();
        int sum = 0;
        for (int i = 0; i < clusterCount; i++) {
            int rg = r.nextInt(50) + 10;
            numberOfDataInEachArea[i] = (rg);
            sum += rg;
        }
        for (int i = 0; i < clusterCount; i++)
            numberOfDataInEachArea[i] = (int)((((double)numberOfDataInEachArea[i]) / sum) * numberOfData);

        Random fRandom = new Random();
        for (int i = 0; i < clusterCount; i++) {
            for (int j = 0; j < numberOfDataInEachArea[i]; j++) {
                ArrayList<Float> tmp = new ArrayList<Float>();
                for (int k = 0; k < dimension; k++) {
                    tmp.add((float)(centroids.get(k).get(i) + fRandom.nextGaussian() * variance));
                }
                data.add(tmp);
            }
        }
    }

    /**
     * Method ini akan melakukan inisialisasi derajat keanggotaan secara acak
     */
    private void assignInitialMembership(){
        u = new float[data.size()][clusterCount];
        u_pre = new float[data.size()][clusterCount];
        Random r = new Random();
        for (int i = 0; i < data.size(); i++) {
            float sum = 0;
            for (int j = 0; j < clusterCount; j++) {
                u[i][j] = r.nextFloat() * 10 + 1;
                sum += u[i][j];
            }
            for (int j = 0; j < clusterCount; j++) {
                u[i][j] = u[i][j] / sum;
            }
        }
    }

    /**
     * Method yang akan menghitung pusat cluster
     */
    private void calculateClusterCenters(){
        clusterCenters.clear();
        for (int i = 0; i < clusterCount; i++) {
            ArrayList<Float> tmp = new ArrayList<Float>();
            for (int j = 0; j < dimension; j++) {
                float cluster_ij;
                float sum1 = 0;
                float sum2 = 0;
                for (int k = 0; k < data.size(); k++) {
                    double tt = Math.pow(u[k][i], fuzziness);
                    sum1 += tt * data.get(k).get(j);
                    sum2 += tt;
                }
                cluster_ij = sum1/sum2;
                tmp.add(cluster_ij);
            }
            clusterCenters.add(tmp);
        }
    }

    /**
     * Method untuk update derajat keanggotaan
     */
    private void updateMembershipValues(){
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < clusterCount; j++) {
                u_pre[i][j] = u[i][j];
                float sum = 0;
                float upper = euclideanDistance(data.get(i), clusterCenters.get(j));
                for (int k = 0; k < clusterCount; k++) {
                    float lower = euclideanDistance(data.get(i), clusterCenters.get(k));
                    sum += Math.pow((upper/lower), 2/(fuzziness -1));
                }
                u[i][j] = 1/sum;
            }
        }
    }

    /**
     * Method untuk menghitung jarak antara 2 buah titik yang berbeda
     * Formula yang digunakan adalah Euclidean distance
     * @param p1 => titik pertama
     * @param p2 => titik kedua
     * @return
     */
    private float euclideanDistance(ArrayList<Float> p1, ArrayList<Float> p2){
        float sum = 0;
        for (int i = 0; i < p1.size(); i++) {
            sum += Math.pow(p1.get(i) - p2.get(i), 2);
        }
        sum = (float) Math.sqrt(sum);
        return sum;
    }

    /**
     * Method ini akan menghitung konvergenitas dari derajat keanggotaan yang lama dan yang baru
     * @return
     */
    private double checkConvergence(){
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < clusterCount; j++) {
                sum += Math.pow(u[i][j] - u_pre[i][j], 2);
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * Method untuk menulis data ke file
     * File yang digunakan berformat .csv
     * @throws IOException
     */
    public void writeDataToFile(ArrayList<ArrayList<Float>> inpData, String fileName) throws IOException {

        FileWriter fileWriter = new FileWriter("./" + fileName + ".csv");
        PrintWriter printWriter = new PrintWriter(fileWriter);

        for (int i = 0; i < inpData.size(); i++) {
            String res = "";
            for (int j = 0; j < inpData.get(i).size(); j++) {
                if(j == inpData.get(i).size() - 1)
                    res += inpData.get(i).get(j);
                else
                    res += inpData.get(i).get(j) +",";
            }
            printWriter.println(res);
        }
        printWriter.close();
    }

    /**
     * Method untuk menulis data cluster ke file
     * File yang digunakan berformat .csv
     * @throws IOException
     */
    public void writeClusterToFile(ArrayList<ArrayList<Float>> inpData, String fileName) throws IOException {

        FileWriter fileWriterCluster = new FileWriter("./" + fileName + ".csv");
        PrintWriter printWriter = new PrintWriter(fileWriterCluster);

        for (int i = 0; i < inpData.size(); i++) {
            String res = "";
            for (int c = 0; c < clusterCount; c++) {
                if (u[i][c] > 0.5) {
                    for (int j = 0; j < inpData.get(i).size(); j++) {
                        if(j == inpData.get(i).size() - 1)
                            res += inpData.get(i).get(j);
                        else
                            res += inpData.get(i).get(j) +",";
                    }
                    printWriter.println(res+",c"+Integer.toString(c+1));
                }
            }
        }
        printWriter.close();
    }


}