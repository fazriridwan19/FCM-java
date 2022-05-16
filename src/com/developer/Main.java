package com.developer;

import java.io.IOException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws IOException {
        Scanner input = new Scanner(System.in);
        Scanner sc = new Scanner(System.in);
        FCM cmean = new FCM("data_lulus_tepat_waktu.csv", 4, 0.0001);

        System.out.print("Number of cluster : ");
        String read1 = sc.nextLine();

        while (true) {
            System.out.println("\n[1] Training");
            System.out.println("[2] Testing");
            System.out.println("[3] Exit");
            System.out.print("Pilih : ");
            int pilihan = input.nextInt();
            if (pilihan == 1) {
                try {
                    cmean.run(Integer.parseInt(read1), 100);
                    System.out.println("Clustering Finished!!!");
                } catch (IOException e) {
                    System.err.println(e);
                }
            } else if (pilihan == 2) {
                try {
                    cmean.runTest("data_lulus_tepat_waktu_test.csv");
                    System.out.println("Clustering Finished!!!");
                } catch (IOException e) {
                    System.err.println(e);
                }
            } else {
                break;
            }
        }
    }
}