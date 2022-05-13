package com.developer;

import java.io.*;
import java.util.*;

public class FCM {
    public ArrayList<ArrayList<Float>> data;
    public ArrayList<ArrayList<Float>> clusterCenters;
    private float u[][];
    private float u_pre[][];
    private int clusterCount;
    private int epochs;
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

    public void run(int clusterNumber, int epochs, ArrayList<ArrayList<Float>> data){
        this.clusterCount = clusterNumber;
        this.epochs = epochs;
        this.data = data;

        // Algoritma FCM
        // 1 Inisialisasi derajat keanggotaan
        assignInitialMembership();

        for (int i = 0; i < epochs; i++) {
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
     * Method ini akan membaca data yang telah ditentukan
     */
    public void readData() {
        String line = "";
        String delim = ",";
        float value;
        ArrayList<ArrayList<String>> temp = new ArrayList<ArrayList<String>>();
        try {
            // Baca data yang ada di file csv lalu simpan ke variable temp
            BufferedReader br = new BufferedReader(new FileReader("data_lulus_tepat_waktu.csv"));
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
                data.add(floatList);
            }
            this.dimension = data.get(0).size();
        }
        catch(IOException e) {
            e.printStackTrace();
        }

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
