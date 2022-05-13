package com.developer;

import java.io.IOException;
import java.sql.SQLOutput;
import java.util.Scanner;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;

public class Main {

    public static void main(String[] args) throws IOException, CsvException {
        FCM cmean = new FCM();

        //get number of class from user
        System.out.println("Please input number of cluster that you want :");
        Scanner sc= new Scanner(System.in);
        String read1 = sc.nextLine();

        //read data
        cmean.readData();

        //run clustering algorithm
        cmean.run(Integer.parseInt(read1), 100, cmean.data);

        //write cluster center to file
        cmean.writeDataToFile(cmean.clusterCenters, "cluster_centers");

        //write data cluster to file
        cmean.writeClusterToFile(cmean.data, "data_cluster");

        System.out.println("Clustering Finished!!!");


    }
}