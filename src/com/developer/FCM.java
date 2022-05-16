package com.developer;

import java.io.*;
import java.util.*;

public class FCM {
    public ArrayList<ArrayList<Float>> data;
    public ArrayList<ArrayList<Float>> clusterCenters;
    private float[][] u;
    private float[][] u_pre;
    private int clusterCount;
    private int dimension;
    private final double fuzziness;
    private final double epsilon;
    public double error;
    public double rsse;

    public FCM(String fileName, double fuzzinessParam, double epsilon){
        this.data = readData(fileName);
        this.clusterCenters = new ArrayList<ArrayList<Float>>();
        this.fuzziness = fuzzinessParam;
        this.epsilon = epsilon;
    }

    public void run(int clusterNumber, int epochs) throws IOException {
        this.clusterCount = clusterNumber;

        // Algoritma FCM
        // 1 Inisialisasi derajat keanggotaan
        System.out.println("============== Training ==============");
        assignInitialMembership(this.data.size(), this.clusterCount);

        for (int i = 0; i < epochs; i++) {
            // 2 Hitung pusat cluster
            calculateClusterCenters();

            // 3 Update derajat keanggotaan
            updateMembershipValues(this.data);

            // 4 Cek konvergensi
            this.error = checkConvergence(this.data);
            this.rsse = rootSumSquaredError(this.data);
            System.out.println(i + ". Konvergensi :" + this.error + " RSSE : " + this.rsse);

            System.out.println("");
            if(this.error <= epsilon)
                break;
        }
        System.out.println("Final RSSE training : " + this.rsse);
        System.out.println("Nilai konvergensi training : " + this.error);
        //write cluster center to file
        this.writeDataToFile(this.clusterCenters, "cluster_centers");
        //write data cluster to file
        this.writeClusterToFile(this.data, "data_cluster");
    }

    public void runTest(String fileName) throws IOException {
        ArrayList<ArrayList<Float>> dataTest;
        dataTest = this.readData(fileName);

        assignInitialMembership(dataTest.size(), clusterCount);
        updateMembershipValues(dataTest);

        this.rsse = rootSumSquaredError(dataTest);

        this.writeClusterToFile(dataTest, "datatest_cluster");
        System.out.println("Final RSSE test : " + this.rsse);
        System.out.println("Test Successfully");
    }

    /**
     * Method ini akan melakukan inisialisasi derajat keanggotaan secara acak
     */
    private void assignInitialMembership(int size, int cluster){
        this.u = new float[size][cluster];
        this.u_pre = new float[size][cluster];
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            float sum = 0;
            for (int j = 0; j < cluster; j++) {
                this.u[i][j] = r.nextFloat() * 10 + 1;
                sum += this.u[i][j];
            }
            for (int j = 0; j < cluster; j++) {
                this.u[i][j] = this.u[i][j] / sum;
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
                    double tt = Math.pow(u[k][i], this.fuzziness);
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
    private void updateMembershipValues(ArrayList<ArrayList<Float>> data){
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < clusterCount; j++) {
                this.u_pre[i][j] = this.u[i][j];
                float sum = 0;
                float upper = euclideanDistance(data.get(i), this.clusterCenters.get(j));
                for (int k = 0; k < this.clusterCount; k++) {
                    float lower = euclideanDistance(data.get(i), this.clusterCenters.get(k));
                    sum += Math.pow((upper/lower), 2.0/(this.fuzziness - 1));
                }
                this.u[i][j] = 1/sum;
            }
        }
    }

    /**
     * Method untuk menghitung jarak antara 2 buah titik yang berbeda
     * Formula yang digunakan adalah Euclidean distance
     * @param p1 => titik pertama
     * @param p2 => titik kedua
     * @return sum
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
     * @return result
     */
    private double checkConvergence(ArrayList<ArrayList<Float>> data){
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < clusterCount; j++) {
                sum += Math.pow(u[i][j] - u_pre[i][j], 2);
            }
        }
        return Math.sqrt(sum);
    }

    private double rootSumSquaredError(ArrayList<ArrayList<Float>> data) {
        float result = 0;
        float dist;

        for (int j = 0; j < this.clusterCount; j++) {
            for (int i = 0; i < data.size(); i++) {
                dist = euclideanDistance(data.get(i), this.clusterCenters.get(j));
                result += Math.pow(u[i][j], this.fuzziness) * Math.pow(dist, 2);
            }
        }

        return Math.sqrt(result);
    }

    /**
     * Method ini akan membaca data yang telah ditentukan
     */
    public ArrayList<ArrayList<Float>> readData(String fileName) {
        String line;
        String delim = ",";
        float value;
        ArrayList<ArrayList<String>> temp = new ArrayList<ArrayList<String>>();
        ArrayList<ArrayList<Float>> result = new ArrayList<ArrayList<Float>>();
        try {
            // Baca data yang ada di file csv lalu simpan ke variable temp
            BufferedReader br = new BufferedReader(new FileReader(fileName));
            while ((line = br.readLine()) != null) {
                String[] csvData = line.split(delim);
                ArrayList<String> al = new ArrayList<String>(Arrays.asList(csvData));
                al.remove(4); // Menghapus index ke-4 atau kolom terakhir pada dataset, yaitu label untuk setiap data
                temp.add(al);
            }
            temp.remove(0); // Menghapus baris pertama atau header dari dataset

            for (int i = 0; i < temp.size(); i++) {
                ArrayList<Float> floatList = new ArrayList<Float>();
                for (int j = 0; j < temp.get(i).size(); j++) {
                    value = Float.parseFloat(temp.get(i).get(j));
                    floatList.add(value);
                }
                result.add(floatList);
            }
            this.dimension = result.get(0).size();
            br.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    /**
     * Method untuk menulis data ke file
     * File yang digunakan berformat .csv
     */
    public void writeDataToFile(ArrayList<ArrayList<Float>> inpData, String fileName) throws IOException {

        FileWriter fileWriter = new FileWriter("./" + fileName + ".csv");
        PrintWriter printWriter = new PrintWriter(fileWriter);

        for (int i = 0; i < inpData.size(); i++) {
            StringBuilder res = new StringBuilder();
            for (int j = 0; j < inpData.get(i).size(); j++) {
                if(j == inpData.get(i).size() - 1)
                    res.append(inpData.get(i).get(j));
                else
                    res.append(inpData.get(i).get(j)).append(",");
            }
            printWriter.println(res);
        }
        printWriter.close();
    }

    /**
     * Method untuk menulis data cluster ke file
     * File yang digunakan berformat .csv
     */
    public void writeClusterToFile(ArrayList<ArrayList<Float>> inpData, String fileName) throws IOException {

        FileWriter fileWriterCluster = new FileWriter("./" + fileName + ".csv");
        PrintWriter printWriter = new PrintWriter(fileWriterCluster);

        for (int i = 0; i < inpData.size(); i++) {
            StringBuilder res = new StringBuilder();
            for (int c = 0; c < clusterCount; c++) {
                if (u[i][c] > 0.5) {
                    for (int j = 0; j < inpData.get(i).size(); j++) {
                        if(j == inpData.get(i).size() - 1)
                            res.append(inpData.get(i).get(j));
                        else
                            res.append(inpData.get(i).get(j)).append(",");
                    }
                    printWriter.println(res+",c"+ (c + 1));
                }
            }
        }
        printWriter.close();
    }


}